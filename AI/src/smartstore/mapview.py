"""Presentation-grade rendering of the geospatial store map.

Draws the measured store model plus the continuous density field from
`heatmap.DensityField`. Everything geometric here (floor outline, zone
polygons, person markers) is drawn as vectors at render resolution so it stays
crisp; only the heat raster is scaled up from the density field's own sampling
grid, which is what keeps it smooth and cheap.

Layer order, back to front:

    structure  ->  heat  ->  floor outline  ->  zones  ->  hotspot
               ->  people  ->  chrome (title, scale, legend)
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .floormap import FloorMap
from .geometry import nice_scale_length
from .heatmap import DensityField, colormap_strip

BG = (18, 16, 14)
FG = (238, 238, 232)
MUTED = (150, 148, 142)
GRID_LINE = (44, 42, 40)
PERSON = (96, 255, 150)

# Tinted structure backdrops, keyed by (w, h, show_structure).
_BACKDROP: dict = {}

# The legend is just the colour scale, so it is built once instead of
# re-colour-mapping the whole density field every frame.
_LEGEND_RAMP = colormap_strip()


def _hsv(h: float, sat: float = 0.55, val: float = 1.0) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, sat, val)
    return (int(b * 255), int(g * 255), int(r * 255))


def alpha_poly(dst: np.ndarray, poly: np.ndarray, color: tuple[int, int, int],
               alpha: float, line: tuple[int, int, int] | None = None,
               line_w: int = 1) -> None:
    """Fill a polygon translucently, then optionally outline it.

    Work is confined to the polygon's bounding box, and the blend runs through
    cv2.addWeighted plus a masked copy. Doing the same thing with boolean fancy
    indexing on the full canvas costs several milliseconds per zone, which is
    more than the rest of the map put together.
    """
    if len(poly) < 3:
        return
    H, W = dst.shape[:2]
    bx0 = int(max(poly[:, 0].min(), 0))
    bx1 = int(min(poly[:, 0].max() + 1, W))
    by0 = int(max(poly[:, 1].min(), 0))
    by1 = int(min(poly[:, 1].max() + 1, H))
    if bx0 >= bx1 or by0 >= by1:
        return
    region = dst[by0:by1, bx0:bx1]
    local = poly - np.array([bx0, by0], np.int32)
    mask = np.zeros(region.shape[:2], np.uint8)
    cv2.fillPoly(mask, [local], 255)
    if alpha >= 0.999:
        cv2.copyTo(np.full_like(region, color), mask, region)
    else:
        solid = np.empty_like(region)
        solid[:] = color
        blended = cv2.addWeighted(solid, alpha, region, 1.0 - alpha, 0)
        cv2.copyTo(blended, mask, region)
    if line is not None and line_w > 0:
        cv2.polylines(dst, [poly], True, line, line_w, cv2.LINE_AA)


def _text(img, s, org, scale, color, weight=1, shadow=True):
    if shadow:
        cv2.putText(img, s, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), weight + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _label_with_box(img, s, org, scale, fg, bg, weight=1):
    (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
    pad = 4
    x, y = org
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base - pad), bg, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, weight, cv2.LINE_AA)


def hotspot_label(fmap: FloorMap, hot: dict | None) -> str:
    """Prefer the semantic zone name; fall back to raw coordinates."""
    if not hot:
        return "-"
    p = np.array([[hot["x_m"], hot["y_m"]]], dtype=np.float32)
    for zone in fmap.zones:
        if zone.kind == "grid":
            continue
        if zone.contains(p)[0]:
            return zone.label or zone.id
    return f"x={hot['x_m']:.1f}m y={hot['y_m']:.1f}m"


def floor_size(fmap, size: tuple[int, int] | None = None,
               px_per_m: float | None = None) -> tuple[int, int]:
    """The (width, height) render_map will produce for these arguments.

    Callers that render every frame use this to allocate the canvas once.
    """
    if px_per_m is None:
        px_per_m = 40.0 if size is None else min(size[0] / max(fmap.width_m, 1e-6),
                                                  size[1] / max(fmap.height_m, 1e-6))
    px = max(float(px_per_m), 1.0)
    return (max(int(round(fmap.width_m * px)), 8),
            max(int(round(fmap.height_m * px)), 8))


def render_map(
    fmap: FloorMap,
    field: DensityField | None = None,
    points: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    trails: dict | None = None,
    occupancy: list[int] | None = None,
    hot: dict | None = None,
    size: tuple[int, int] | None = None,
    px_per_m: float | None = None,
    cumulative: bool = False,
    heat_alpha: float = 0.78,
    show_structure: bool = True,
    show_grid: bool = False,
    show_zone_labels: bool = True,
    show_dimensions: bool = True,
    show_occupancy: bool = True,
    show_track_ids: bool = True,
    show_trails: bool = True,
    show_hotspot: bool = True,
    show_scale_bar: bool = True,
    show_legend: bool = True,
    title: str = "REAL-TIME SPATIAL ACTIVITY",
    canvas: np.ndarray | None = None,
) -> np.ndarray:
    """Render the store map. Returns a BGR image of the floor only.

    `size` fits the floor into a box; `px_per_m` overrides the scale outright.
    Pass `canvas` sized by `floor_size` to draw into a buffer you already own.
    """
    if px_per_m is None:
        if size is None:
            px_per_m = 40.0
        else:
            px_per_m = min(size[0] / max(fmap.width_m, 1e-6),
                           size[1] / max(fmap.height_m, 1e-6))
    px = max(float(px_per_m), 1.0)
    W = max(int(round(fmap.width_m * px)), 8)
    H = max(int(round(fmap.height_m * px)), 8)
    k = max(px / 40.0, 0.55)
    if canvas is None or canvas.shape != (H, W, 3):
        canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = BG

    def P(pts_m: np.ndarray) -> np.ndarray:
        a = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
        return np.stack([a[:, 0] * px, a[:, 1] * px], axis=1).round().astype(np.int32)

    # ---- measured geometry: the rectified store -------------------------
    # The tinted structure is identical every frame, so it is built once per
    # output size and copied. Tinting costs ~2ms at panel resolution, which
    # would otherwise be paid 25 times a second for a constant image.
    key = (W, H, show_structure)
    backdrop = _BACKDROP.get(key)
    if backdrop is None:
        backdrop = np.full((H, W, 3), BG, dtype=np.uint8)
        base = fmap._load_structure_layer((W, H)) if show_structure else None
        if base is not None:
            # Cool and darken so the heat colours on top stay dominant and the
            # shelves read as context rather than as data.
            g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            cool = np.stack(
                [(g * 0.62).astype(np.uint8), (g * 0.72).astype(np.uint8),
                 (g * 0.80).astype(np.uint8)], axis=2)
            backdrop = cv2.addWeighted(cool, 0.42, backdrop, 0.58, 0)
        _BACKDROP[key] = backdrop
    canvas = backdrop.copy()

    # ---- continuous density field --------------------------------------
    # One peak evaluation per frame: it is stateful (smoothed), so calling it
    # twice would decay the reference value twice as fast.
    top = field.peak(cumulative) if (field is not None and field.enabled) else 0.0
    if field is not None and field.enabled:
        heat = field.image(cumulative=cumulative, vmax=top) if top > 0 else None
        if heat is not None:
            up = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
            if field.mask_is_full:
                # Rectangular store: one SIMD blend instead of three 2.5M-element
                # float temporaries. This is the difference between 0.4ms and 5.4ms.
                canvas = cv2.addWeighted(up, heat_alpha, canvas, 1.0 - heat_alpha, 0)
            else:
                # Non-rect floor: rescale the mask with the raster so the sales
                # area still has a hard edge after upsampling.
                m = np.clip(cv2.resize(field.mask, (W, H),
                                       interpolation=cv2.INTER_LINEAR) * 1.6, 0.0, 1.0)
                a = (m * heat_alpha)[:, :, None]
                canvas = (canvas.astype(np.float32) * (1.0 - a)
                          + up.astype(np.float32) * a).astype(np.uint8)

    if show_grid:
        step = 1.0
        n = int(fmap.width_m / step) + 1
        for i in range(1, n):
            cv2.line(canvas, (int(i * step * px), 0), (int(i * step * px), H),
                     GRID_LINE, 1, cv2.LINE_AA)
        n = int(fmap.height_m / step) + 1
        for j in range(1, n):
            cv2.line(canvas, (0, int(j * step * px)), (W, int(j * step * px)),
                     GRID_LINE, 1, cv2.LINE_AA)

    # ---- sales-floor outline -------------------------------------------
    fp = fmap.floor_polygon
    outline = P(np.asarray(fp if fp else fmap.bounds, dtype=np.float64))
    cv2.polylines(canvas, [outline], True, (90, 200, 250), max(1, int(round(1.6 * k))),
                  cv2.LINE_AA)

    # ---- semantic zones -------------------------------------------------
    occ = occupancy or []
    for i, zone in enumerate(fmap.zones):
        if zone.kind == "grid":
            continue
        count = occ[i] if i < len(occ) else 0
        m = zone.measures
        # Heat-driven hue: an empty zone stays cool, an occupied one goes warm.
        frac = min(count / 4.0, 1.0)
        base_hue = 0.55 - 0.55 * frac            # cyan -> orange
        fill = _hsv(base_hue, sat=0.42, val=0.95)
        edge = _hsv(base_hue, sat=0.75, val=1.0)
        poly = P(zone.polygon)
        alpha_poly(canvas, poly, fill, 0.20 if count else 0.10, edge,
                   max(1, int(round(1.4 * k))))

        if not show_zone_labels:
            continue
        cx, cy = P(np.array([[m["centroid_x_m"], m["centroid_y_m"]]]))[0]
        lines = [zone.label or zone.id]
        sub = ""
        if show_dimensions:
            sub += f"{m['width_m']:.1f} x {m['height_m']:.1f} m"
        if show_occupancy:
            sub += (f"  |  {count} in" if sub else f"{count} in")
        y = int(cy - 6 * k) if sub else int(cy + 4 * k)
        _label_with_box(canvas, lines[0], (int(cx), y), 0.42 * k, FG, (10, 12, 14), 1)
        if sub:
            _text(canvas, sub, (int(cx), int(cy + 10 * k)), 0.36 * k, MUTED, 1)

    # ---- hotspot --------------------------------------------------------
    if show_hotspot and hot:
        hx, hy = P(np.array([[hot["x_m"], hot["y_m"]]]))[0]
        r = max(int(round(9 * k)), 4)
        cv2.circle(canvas, (hx, hy), r, (60, 90, 255), max(1, int(round(1.4 * k))),
                   cv2.LINE_AA)
        cv2.circle(canvas, (hx, hy), max(r - int(round(3 * k)), 1),
                   (60, 90, 255), 1, cv2.LINE_AA)
        _text(canvas, "HOTSPOT", (hx + r + 4, hy + 4), 0.36 * k, (90, 120, 255), 1)

    # ---- people ---------------------------------------------------------
    if points is not None and len(points):
        ppx = P(points)
        if show_trails and trails:
            for hist in trails.values():
                if len(hist) < 2:
                    continue
                hp = P(np.asarray(hist, dtype=np.float64))
                for a, b in zip(hp[:-1], hp[1:]):
                    cv2.line(canvas, tuple(a), tuple(b), (70, 150, 90),
                             max(1, int(round(1.1 * k))), cv2.LINE_AA)
        rr = max(int(round(4.2 * k)), 2)
        for i, (x, y) in enumerate(ppx):
            cv2.circle(canvas, (int(x), int(y)), rr, PERSON, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), rr, (10, 40, 20),
                       max(1, int(round(1.0 * k))), cv2.LINE_AA)
            if show_track_ids and labels is not None and i < len(labels):
                _text(canvas, str(int(labels[i])), (int(x) + rr + 2, int(y) - rr - 1),
                      0.34 * k, (170, 255, 190), 1)

    # ---- chrome ---------------------------------------------------------
    if title:
        _text(canvas, title, (int(8 * k), int(16 * k)), 0.46 * k, FG, 1)

    if show_scale_bar and px_per_m is None and size is not None:
        metres = nice_scale_length(fmap.width_m)
        bar = int(metres * px)
        y = H - int(10 * k)
        x0 = int(8 * k)
        cv2.line(canvas, (x0, y), (x0 + bar, y), FG, max(1, int(round(1.6 * k))),
                 cv2.LINE_AA)
        cv2.line(canvas, (x0, y - int(3 * k)), (x0, y + int(3 * k)), FG, 1, cv2.LINE_AA)
        cv2.line(canvas, (x0 + bar, y - int(3 * k)), (x0 + bar, y + int(3 * k)), FG, 1,
                 cv2.LINE_AA)
        _text(canvas, f"{metres:g} m", (x0 + bar + int(5 * k), y + int(4 * k)),
              0.36 * k, MUTED, 1)

    if show_legend and top > 0:
        lw = int(min(W * 0.32, 160 * k))
        lh = max(int(8 * k), 6)
        lx, ly = W - lw - int(12 * k), H - lh - int(24 * k)
        # Index the 256-entry ramp directly: exact, and cheaper than a resize.
        cols = _LEGEND_RAMP[np.linspace(0, 255, lw).round().astype(np.int32)]
        canvas[ly:ly + lh, lx:lx + lw] = np.repeat(cols[None, :, :], lh, axis=0)
        cv2.rectangle(canvas, (lx - 1, ly - 1), (lx + lw + 1, ly + lh + 1),
                      (70, 68, 64), 1)
        _text(canvas, "LOW", (lx - 1, ly - int(4 * k)), 0.32 * k, MUTED, 1)
        _text(canvas, "HIGH", (lx + lw - int(22 * k), ly - int(4 * k)),
              0.32 * k, MUTED, 1)
        _text(canvas, "CUMULATIVE" if cumulative else "RECENT",
              (lx, ly + lh + int(10 * k)), 0.3 * k, MUTED, 1)

    return canvas
