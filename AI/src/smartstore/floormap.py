"""Floor geometry and 2D map rendering.

One homography maps camera pixels onto the floor plane in metres. Everything
else — zone lookup, footfall, rendering — is built on that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import polygon_measures

DEFAULT_PX_PER_M = 16.0
MIN_REFERENCE_AREA_PX = 400.0

JET = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8).reshape(1, 256), cv2.COLORMAP_JET
).reshape(256, 3)


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd test. pts (M,2) -> bool (M,).

    Uses a half-open rule (`x <= x_cross`) so points sitting on a zone edge are
    counted rather than flickering in and out of the count.
    """
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    straddles = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return (straddles & (x <= x_cross)).sum(axis=1) % 2 == 1


def heat_colors(values: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Per-cell BGR colours for an accumulated density layer.

    A sqrt curve is used because a single dwell hotspot would otherwise wash
    every other cell to the same colour. Returns an (M,1,3) uint8 array ready
    for cv2.fillConvexPoly.
    """
    v = np.asarray(values, dtype=np.float64)
    top = float(vmax if vmax is not None else v.max(initial=0.0))
    if top <= 0:
        return np.zeros((len(v), 1, 3), np.uint8)
    u8 = np.clip(np.sqrt(v / top) * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    return cv2.applyColorMap(u8, cmap)


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of a polygon."""
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def rect_polygon(width: float, height: float) -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float32
    )


def _zone_color(fraction: float) -> tuple[int, int, int]:
    """Heat colour for a 0..1 density fraction."""
    idx = int(np.clip(fraction, 0.0, 1.0) * 255)
    b, g, r = (int(v) for v in JET[idx])
    return (b, g, r)


@dataclass
class Zone:
    id: str
    kind: str
    polygon: np.ndarray
    label: str = ""

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)

    @property
    def measures(self) -> dict:
        """Real-world area/width/height in metres, cached per zone.

        Zone polygons are authored in floor metres (`map.zones`), so these are
        direct measurements of the mapped plane, not estimates from pixels.
        """
        cached = self.__dict__.get("_measures")
        if cached is None:
            cached = polygon_measures(self.polygon, self.id, self.kind)
            self.__dict__["_measures"] = cached
        return cached

    def contains(self, pts: np.ndarray) -> np.ndarray:
        return points_in_polygon(pts, self.polygon)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label or self.id,
            "polygon": [[round(float(a), 4), round(float(b), 4)] for a, b in self.polygon],
        }
        # Geometry travels with the zone so the scanner never has to re-derive
        # it, and downstream consumers get dimensions straight from layout.json.
        if self.kind != "grid":
            d.update(self.measures)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Zone:
        return cls(
            id=d["id"],
            kind=d.get("kind", "area"),
            polygon=np.array(d["polygon"], dtype=np.float32),
            label=d.get("label", ""),
        )


def build_zones(
    width_m: float, height_m: float, grid_m: float, named: list[dict] | None
) -> list[Zone]:
    """Named zones first (they win on overlap), then the uniform metric grid."""
    named_zones = [Zone.from_dict(d) for d in named or []]
    grid: list[Zone] = []
    cell = max(float(grid_m), 0.25)
    for row, y0 in enumerate(np.arange(0.0, height_m, cell)):
        for col, x0 in enumerate(np.arange(0.0, width_m, cell)):
            x1, y1 = min(x0 + cell, width_m), min(y0 + cell, height_m)
            grid.append(
                Zone(
                    id=f"g{row}_{col}",
                    kind="grid",
                    polygon=np.array(
                        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
                    ),
                    label=f"{x0:.0f},{y0:.0f}",
                )
            )
    return named_zones + grid


@dataclass
class FloorMap:
    homography: np.ndarray
    width_m: float
    height_m: float
    zones: list[Zone] = field(default_factory=list)
    boundary_margin_m: float = 0.15
    structure_path: str | None = None
    structure_gamma: float = 0.85
    px_per_m: float = DEFAULT_PX_PER_M
    roi_polygon: list[list[float]] = field(default_factory=list)
    # Sales-floor outline in metres. None -> the calibrated width_m x height_m
    # rect. A non-rect floor lets the heat layer be clipped to real geometry.
    floor_polygon: list | None = None
    meta: dict = field(default_factory=dict)
    # Keyed by (w, h): the map video and the preview render at different
    # resolutions in the same run, and a single cached layer would mismatch
    # the canvas in cv2.addWeighted.
    _structure_cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ---- construction -------------------------------------------------

    @classmethod
    def from_reference_points(
        cls,
        points,
        width_m: float,
        height_m: float,
        grid_m: float = 1.0,
        boundary_margin_m: float = 0.15,
        named_zones: list[dict] | None = None,
        structure_path: str | None = None,
        structure_gamma: float = 0.85,
        px_per_m: float = DEFAULT_PX_PER_M,
        roi_polygon: list | None = None,
        meta: dict | None = None,
    ) -> FloorMap:
        if len(points) != 4:
            raise ValueError("need exactly 4 reference points")
        quad = np.array(points, dtype=np.float32)
        # Collinear or near-duplicate points make getPerspectiveTransform emit a
        # silently degenerate homography, so reject them up front.
        if polygon_area(quad) < MIN_REFERENCE_AREA_PX:
            raise ValueError(
                "reference points form a degenerate area - click four distinct "
                "floor corners forming a rectangle, not a straight line"
            )
        w_px, h_px = width_m * px_per_m, height_m * px_per_m
        dst = np.array([[0.0, 0.0], [w_px, 0.0], [w_px, h_px], [0.0, h_px]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(quad, dst)
        roi = [[float(x), float(y)] for x, y in (roi_polygon or [])]
        if roi and len(roi) < 3:
            raise ValueError("roi_polygon needs at least 3 points")
        return cls(
            homography=H,
            width_m=width_m,
            height_m=height_m,
            px_per_m=px_per_m,
            roi_polygon=roi,
            zones=build_zones(width_m, height_m, grid_m, named_zones),
            boundary_margin_m=boundary_margin_m,
            structure_path=structure_path,
            structure_gamma=structure_gamma,
            meta=meta or {},
        )

    # ---- geometry -----------------------------------------------------

    @property
    def bounds(self) -> np.ndarray:
        return rect_polygon(self.width_m, self.height_m)

    def to_floor(self, pts: np.ndarray) -> np.ndarray:
        """Camera pixels (M,2) -> floor metres (M,2).

        The homography targets a pixel grid of `px_per_m` pixels per metre, so
        the projective divide is followed by a scale-down to true metres.
        """
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        ones = np.ones((len(pts), 1), dtype=np.float32)
        stacked = np.hstack([pts.astype(np.float32), ones]).T
        out = self.homography @ stacked
        w = np.where(np.abs(out[2]) < 1e-9, 1e-9, out[2])
        return (np.vstack([out[0] / w, out[1] / w]).T / self.px_per_m).astype(np.float32)

    def inside(self, pts: np.ndarray) -> np.ndarray:
        return points_in_polygon(pts, self.bounds)

    def in_roi(self, pts: np.ndarray) -> np.ndarray:
        """True for camera-pixel points inside the ROI polygon.

        An empty ROI means "no restriction" and returns all True, so the
        feature is opt-in and cannot silently zero out a run.
        """
        if not self.roi_polygon or len(pts) == 0:
            return np.ones(len(pts), dtype=bool)
        return points_in_polygon(pts, np.asarray(self.roi_polygon, dtype=np.float32))

    def zone_index_for(self, pts: np.ndarray) -> np.ndarray:
        """First containing zone per point, -1 when outside all zones. (M,)"""
        out = np.full(len(pts), -1, dtype=np.int32)
        if len(pts) == 0:
            return out
        for idx, zone in enumerate(self.zones):
            hit = zone.contains(pts) & (out < 0)
            out[hit] = idx
        return out

    def occupancy(self, pts: np.ndarray) -> list[int]:
        idx = self.zone_index_for(pts)
        counts = [0] * len(self.zones)
        for z in np.unique(idx[idx >= 0]):
            counts[int(z)] = int((idx == z).sum())
        return counts

    # ---- serialisation ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "homography": [[round(float(v), 10) for v in row] for row in self.homography],
            "width_m": self.width_m,
            "height_m": self.height_m,
            "px_per_m": self.px_per_m,
            "roi_polygon": self.roi_polygon,
            "floor_polygon": ([[float(x), float(y)] for x, y in self.floor_polygon]
                             if self.floor_polygon else None),
            "boundary_margin_m": self.boundary_margin_m,
            "structure_path": self.structure_path,
            "structure_gamma": self.structure_gamma,
            "zones": [z.to_dict() for z in self.zones],
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def from_dict(cls, d: dict) -> FloorMap:
        return cls(
            homography=np.array(d["homography"], dtype=np.float64),
            width_m=float(d["width_m"]),
            height_m=float(d["height_m"]),
            px_per_m=float(d.get("px_per_m", DEFAULT_PX_PER_M)),
            roi_polygon=[[float(x), float(y)] for x, y in d.get("roi_polygon") or []],
            floor_polygon=([list(map(float, p)) for p in d["floor_polygon"]]
                            if d.get("floor_polygon") else None),
            zones=[Zone.from_dict(z) for z in d["zones"]],
            boundary_margin_m=float(d.get("boundary_margin_m", 0.15)),
            structure_path=d.get("structure_path"),
            structure_gamma=float(d.get("structure_gamma", 0.85)),
            meta=d.get("meta", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> FloorMap:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- rendering ----------------------------------------------------

    def _load_structure_layer(self, size: tuple[int, int]) -> np.ndarray | None:
        cached = self._structure_cache.get(size)
        if cached is not None:
            return cached
        if not self.structure_path or not Path(self.structure_path).exists():
            return None
        img = cv2.imread(str(self.structure_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        warped = cv2.warpPerspective(img, self.homography, size, flags=cv2.INTER_LINEAR)
        lut = np.clip(
            255.0 * np.power(np.arange(256) / 255.0, self.structure_gamma), 0, 255
        ).astype(np.uint8)
        layer = cv2.LUT(warped, lut)
        self._structure_cache[size] = layer
        return layer

    def render(
        self,
        occupancy: list[int] | None = None,
        points: np.ndarray | None = None,
        labels: list[int] | None = None,
        header: str = "",
        px_per_m: float = DEFAULT_PX_PER_M,
        accum: np.ndarray | None = None,
    ) -> np.ndarray:
        W = max(int(self.width_m * px_per_m), 1)
        H = max(int(self.height_m * px_per_m), 1)
        canvas = np.full((H, W, 3), 22, dtype=np.uint8)
        # Glyph sizes below are authored for DEFAULT_PX_PER_M. Scale them with the
        # canvas so a higher-resolution render stays legible instead of shrinking
        # the text. k == 1 reproduces the default rendering exactly.
        k = px_per_m / DEFAULT_PX_PER_M
        thick = max(1, round(1 * k))
        head_h = max(14, round(24 * k))

        base = self._load_structure_layer((W, H))
        if base is not None:
            canvas = cv2.addWeighted(base, 0.85, canvas, 0.15, 0)

        occ = [0] * len(self.zones) if occupancy is None else occupancy
        peak = max(max(occ, default=0), 1)

        # Accumulated person-seconds underneath, so the map shows where time was
        # spent and builds up smoothly instead of flickering frame to frame.
        if accum is not None and len(accum) == len(self.zones):
            for zone, value, color in zip(self.zones, accum, heat_colors(accum)):
                if value <= 0:
                    continue
                cv2.fillConvexPoly(
                    canvas, self._to_canvas(zone.polygon, px_per_m, H),
                    (int(color[0][0]), int(color[0][1]), int(color[0][2])),
                    lineType=cv2.LINE_AA,
                )

        for zone, count in zip(self.zones, occ):
            if count <= 0:
                continue
            poly = self._to_canvas(zone.polygon, px_per_m, H)
            cv2.fillConvexPoly(canvas, poly, _zone_color(count / peak), lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [poly], True, (60, 60, 60), thick, cv2.LINE_AA)

        for zone in self.zones:
            if zone.kind == "grid":
                continue
            poly = self._to_canvas(zone.polygon, px_per_m, H)
            tint = (0, 215, 255) if zone.kind == "queue" else (255, 200, 0)
            cv2.polylines(canvas, [poly], True, tint, max(1, round(2 * k)), cv2.LINE_AA)

        if points is not None and len(points):
            for i, (x, y) in enumerate(self._to_canvas(points, px_per_m, H)):
                cv2.circle(canvas, (int(x), int(y)), max(2, round(4 * k)),
                           (60, 255, 120), -1, cv2.LINE_AA)
                if labels is not None and i < len(labels):
                    cv2.putText(
                        canvas, str(int(labels[i])),
                        (int(x) + round(6 * k), int(y) - round(6 * k)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38 * k, (200, 255, 200),
                        thick, cv2.LINE_AA,
                    )

        for zone, count in zip(self.zones, occ):
            if zone.kind == "grid" or count <= 0:
                continue
            cx, cy = self._to_canvas(zone.polygon.mean(axis=0, keepdims=True), px_per_m, H)[0]
            cv2.putText(
                canvas, f"{zone.label} {count}", (int(cx) - round(30 * k), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, (255, 255, 255), thick, cv2.LINE_AA,
            )

        bar_px = int(px_per_m)
        bar_y = max(10, round(H - 16 * k))
        bar_x, bar_h = round(10 * k), max(4, round(8 * k))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_px, bar_y + bar_h),
                      (255, 255, 255), -1)
        cv2.putText(
            canvas, "1 m", (round(12 * k) + bar_px, bar_y + round(7 * k)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4 * k, (255, 255, 255), thick, cv2.LINE_AA,
        )
        if header:
            cv2.rectangle(canvas, (0, 0), (W, head_h), (0, 0, 0), -1)
            cv2.putText(
                canvas, header, (round(8 * k), round(17 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * k, (255, 255, 255), thick, cv2.LINE_AA,
            )
        return canvas

    @staticmethod
    def _to_canvas(pts: np.ndarray, px_per_m: float, canvas_h: int) -> np.ndarray:
        """Floor metres -> canvas pixels, y down.

        Floor y=0 maps to canvas row 0, matching the homography's destination
        ordering and the `warpPerspective` structure layer. The previous
        `canvas_h - y` flip put the far edge of the store off the bottom of the
        canvas and mirrored the structure image against the analytic overlays.
        """
        out = np.empty_like(pts, dtype=np.int32)
        out[:, 0] = np.round(pts[:, 0] * px_per_m)
        out[:, 1] = np.round(pts[:, 1] * px_per_m)
        return out
