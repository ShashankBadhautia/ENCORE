"""Continuous geospatial density field over the calibrated floor plane.

`FloorMap`'s 1 m grid stays because the existing CSV/JSONL metrics are defined
in terms of it. This module is what the map actually draws: a Gaussian kernel
density estimate sampled on a regular raster, so the surface has no cell edges
and the output is a real spatial field.

Coordinate discipline
---------------------
All geometry in and out of this module is in **floor metres**. The raster is a
private sampling grid whose only job is to evaluate that surface; its
`resolution` is deliberately decoupled from the map's render resolution, so the
density can be computed once at 40 px/m and upscaled smoothly for a large
presentation panel instead of paying for a 4x larger field.

Units
-----
The field holds **person-seconds per square metre**. Each splat distributes a
person's time over a Gaussian whose sigma is `sigma_m` in real units, and the
raster cell area is divided out so a value means the same thing regardless of
resolution.

Cost
----
Splatting touches only a local window around each person, sized by
`max_kernel_px`. Two buffers (a decaying `live` window and a monotonic `total`)
are maintained; both are one float32 add per touched pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .floormap import points_in_polygon, rect_polygon
from .geometry import polygon_area


def colormap() -> int:
    """The density colour scale. Turbo reads better than Jet for ordered data.

    Falls back to Jet on OpenCV builds without Turbo, so the map still renders.
    """
    return cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET


def colormap_strip() -> np.ndarray:
    """(256, 3) BGR ramp for the heat legend, built once."""
    return cv2.applyColorMap(
        np.arange(256, dtype=np.uint8).reshape(1, 256), colormap()
    ).reshape(256, 3)


def retention_for_window(window_seconds: float, dt: float) -> float:
    """Per-second retention giving a half-life of `window_seconds`.

    The exponential decay in `advance` is then applied as `retention ** dt`, so
    the configured window means the same thing at 15 fps and at 60 fps. A value
    of 1.0 disables decay (an unbounded window).
    """
    if window_seconds <= 0 or dt <= 0:
        return 1.0
    return float(0.5 ** (1.0 / window_seconds))


@dataclass
class DensityField:
    """Gaussian KDE of tracked foot points, in person-seconds per m2."""

    width_m: float
    height_m: float
    resolution: float = 40.0
    sigma_m: float = 0.5
    decay: float = 0.95
    max_kernel_px: int = 48
    enabled: bool = True
    boundary: np.ndarray | None = None

    live: np.ndarray = field(init=False, repr=False)
    total: np.ndarray = field(init=False, repr=False)
    mask: np.ndarray = field(init=False, repr=False)
    kernel: np.ndarray = field(init=False, repr=False)
    radius: int = field(init=False, default=0)
    peak_live: float = field(init=False, default=0.0)
    peak_total: float = field(init=False, default=0.0)
    # True when the whole raster is inside the floor, which lets the renderer
    # skip per-pixel mask blending on the common rectangular store.
    mask_is_full: bool = field(init=False, default=True)
    _lut: np.ndarray | None = field(init=False, default=None, repr=False)
    _lut_gamma: float = field(init=False, default=-1.0, repr=False)

    def __post_init__(self) -> None:
        self.resolution = max(float(self.resolution), 1.0)
        self.sigma_m = max(float(self.sigma_m), 0.01)
        self.W = max(int(round(self.width_m * self.resolution)), 1)
        self.H = max(int(round(self.height_m * self.resolution)), 1)
        self.live = np.zeros((self.H, self.W), dtype=np.float32)
        self.total = np.zeros((self.H, self.W), dtype=np.float32)
        self.mask = self._build_mask()
        self.mask_is_full = bool(self.mask.all())
        self.kernel, self.radius = self._build_kernel()

    # ---- construction -------------------------------------------------

    def _build_kernel(self) -> tuple[np.ndarray, int]:
        """Separable-normalised Gaussian stamp, in raster pixels.

        Truncating the tail at 2.5 sigma (or `max_kernel_px`, whichever is
        smaller) keeps the per-person cost bounded at high resolutions. The
        discarded mass is ~1% of the integral, so totals stay meaningful.
        """
        sigma_px = self.sigma_m * self.resolution
        r = int(min(np.ceil(2.5 * sigma_px), max(self.max_kernel_px, 1)))
        r = max(r, 1)
        ax = np.arange(-r, r + 1, dtype=np.float32)
        k = np.exp(-0.5 * (ax / sigma_px) ** 2)
        k /= k.sum()
        return np.outer(k, k).astype(np.float32), r

    def _build_mask(self) -> np.ndarray:
        """1.0 inside the mapped floor, 0.0 outside.

        The 4-point calibration rectifies to a rectangle, so the mask mainly
        clips the Gaussian halo that would otherwise spill past the store edge.
        A custom `map.floor_polygon` (metres) is honoured so a non-rectangular
        sales floor is clipped correctly too.
        """
        m = np.zeros((self.H, self.W), dtype=np.float32)
        poly = self.boundary
        if poly is None or len(poly) < 3:
            poly = rect_polygon(self.width_m, self.height_m)
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, self.width_m)
        pts[:, 1] = np.clip(pts[:, 1], 0.0, self.height_m)
        if polygon_area(pts) <= 0:
            m[:] = 1.0
            return m
        px = np.round(pts * self.resolution).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [px], 1.0)
        return m

    # ---- geometry -----------------------------------------------------

    def to_px(self, pts_m: np.ndarray) -> np.ndarray:
        """Floor metres -> raster pixels, matching FloorMap._to_canvas (y down)."""
        p = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
        out = np.empty((len(p), 2), dtype=np.int32)
        out[:, 0] = np.round(p[:, 0] * self.resolution)
        out[:, 1] = np.round(p[:, 1] * self.resolution)
        return out

    def to_m(self, px: tuple[int, int]) -> tuple[float, float]:
        return (px[0] / self.resolution, px[1] / self.resolution)

    # ---- accumulation -------------------------------------------------

    def splat(self, points_m: np.ndarray, weight: float = 1.0) -> None:
        """Add `weight` person-seconds at each floor point, to live and total.

        The value written is weight distributed over the Gaussian and divided
        by the raster cell area, so the result is person-seconds per m2.
        """
        if not self.enabled or weight <= 0.0 or len(points_m) == 0:
            return
        gain = float(weight) * self.resolution * self.resolution
        if gain <= 0.0:
            return
        r, k = self.radius, self.kernel
        H, W = self.H, self.W
        # The floor mask is all-ones for a normal rect calibration, so skip the
        # per-splat multiply in that case. When a non-rect floor is configured the
        # contribution is clipped at write time, so no mass can accumulate outside
        # the sales area even if a foot point projects there.
        clip = not bool(self.mask.all())
        for x, y in self.to_px(np.asarray(points_m, dtype=np.float64)):
            # A kernel wider than the floor would produce out-of-range slices.
            x0, x1 = max(0, x - r), min(W, x + r + 1)
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            sub = k[y0 - y + r: y1 - y + r, x0 - x + r: x1 - x + r] * gain
            if clip:
                sub = sub * self.mask[y0:y1, x0:x1]
            np.add(self.live[y0:y1, x0:x1], sub, out=self.live[y0:y1, x0:x1])
            np.add(self.total[y0:y1, x0:x1], sub, out=self.total[y0:y1, x0:x1])

    def advance(self, dt: float) -> None:
        """Age the live window by `dt` seconds, re-clipping it to the floor."""
        if dt <= 0.0 or self.decay >= 1.0:
            return
        self.live *= np.float32(self.decay ** dt)
        self.live *= self.mask

    # ---- readouts -----------------------------------------------------

    def _field(self, cumulative: bool) -> np.ndarray:
        return self.total if cumulative else self.live

    def _update_peak(self, cumulative: bool) -> float:
        """Track a smoothed peak so the legend does not strobe.

        A raw max/frame makes the whole map flash between frames whenever the
        busiest pixel changes. The peak falls slowly and rises instantly, so a
        normal frame is stable but a new hotspot still shows up immediately.
        """
        cur = float(self._field(cumulative).max(initial=0.0))
        key = "peak_total" if cumulative else "peak_live"
        prev = getattr(self, key)
        nxt = max(cur, prev * 0.995)
        setattr(self, key, nxt)
        return nxt

    def peak(self, cumulative: bool = False) -> float:
        return self._update_peak(cumulative)

    #: Index resolution of the gamma ramp. 4096 keeps the quantisation error
    #: below one 8-bit level, so the table is visually indistinguishable from
    #: computing the power in float, and it is still cheaper than np.power on
    #: every raster cell.
    LUT_STEPS = 4096

    def _gamma_lut(self, gamma: float) -> np.ndarray:
        """Gamma ramp as a lookup table, cached per gamma.

        Applying gamma through a table instead of np.power on every raster cell
        is worth a few ms a frame at panel resolutions.
        """
        if self._lut is None or self._lut_gamma != gamma:
            n = self.LUT_STEPS
            ramp = np.power(np.arange(n, dtype=np.float32) / (n - 1.0), gamma)
            self._lut = (ramp * 255.0).astype(np.uint8)
            self._lut_gamma = gamma
        return self._lut

    def image(self, cumulative: bool = False, vmax: float | None = None,
              gamma: float = 0.62) -> np.ndarray | None:
        """Colour-mapped BGR density surface, already masked to the floor.

        A gamma below 1 lifts the mid-tones so lightly-used parts of a large
        store are still visible next to one hot spot.
        """
        f = self._field(cumulative)
        top = float(vmax) if vmax else self._update_peak(cumulative)
        if top <= 0.0:
            return None
        n1 = float(self.LUT_STEPS - 1)
        idx = np.clip(f * np.float32(n1 / top), 0.0, n1)
        if not self.mask_is_full:
            idx *= self.mask
        u8 = self._gamma_lut(gamma)[idx.astype(np.uint16)]
        return cv2.applyColorMap(u8, colormap())

    def hotspot(self, cumulative: bool = False) -> dict | None:
        """Densest raster cell, as floor metres, for the HOTSPOT readout."""
        f = self._field(cumulative)
        if not self.mask_is_full:
            f = f * self.mask
        idx = int(np.argmax(f))
        top = float(f.flat[idx])
        if top <= 0.0:
            return None
        y, x = divmod(idx, self.W)
        return {
            "x_m": round(x / self.resolution, 2),
            "y_m": round(y / self.resolution, 2),
            "density": round(top, 3),
            "pixel": (x, y),
        }

    def mean_in(self, polygon_m: np.ndarray) -> float:
        """Mean density (person-seconds per m2) inside a polygon in metres."""
        cells = self.cell_centres()
        if len(cells) == 0:
            return 0.0
        inside = points_in_polygon(cells, np.asarray(polygon_m, dtype=np.float32))
        n = int(inside.sum())
        if n == 0:
            return 0.0
        f = self._field(False)[inside] * self.mask[inside]
        return float(f.sum() / n)

    def cell_centres(self) -> np.ndarray:
        """(H*W, 2) floor-metre centre of every raster cell."""
        if not hasattr(self, "_centres"):
            ys, xs = np.mgrid[0:self.H, 0:self.W]
            self._centres = np.stack(
                [(xs.ravel() + 0.5) / self.resolution,
                 (ys.ravel() + 0.5) / self.resolution], axis=1
            ).astype(np.float32)
        return self._centres

    def person_seconds(self, cumulative: bool = False) -> float:
        """Integrated person-time on the floor, in person-seconds.

        Divides by the raster cell area so the value is the actual quantity of
        shopper-time, independent of the sampling resolution.
        """
        f = self._field(cumulative)
        return float(f.sum() / (self.resolution * self.resolution))

    def mass_outside(self) -> float:
        """Density mass sitting off the floor. Should always be ~0.

        Exposed so a non-rect floor can be regression-checked rather than
        assumed correct.
        """
        outside = ~self.mask.astype(bool)
        if not outside.any():
            return 0.0
        return float(self.total[outside].sum() / (self.resolution * self.resolution))

    @property
    def total_person_seconds(self) -> float:
        return self.person_seconds(cumulative=True)
