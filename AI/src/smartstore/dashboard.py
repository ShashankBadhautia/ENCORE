"""Side-by-side presentation dashboard.

One output stream, built from the same source frame that produced the map, so
video and heatmap cannot drift apart: there is a single read per frame, and the
two panels are composed in the same iteration. No independent timers, no second
decode pass.

    ┌──────────────────────────────────────────────────┐
    │ SMARTSTORE360            LIVE STORE INTELLIGENCE │
    ├────────────────────┬─────────────────────────────┤
    │                    │                             │
    │    CCTV VIDEO      │     REAL-TIME STORE MAP     │
    │                    │                             │
    ├────────────────────┴─────────────────────────────┤
    │ PEOPLE  FOOTFALL  QUEUE  AVG DWELL  HOTSPOT     │
    └──────────────────────────────────────────────────┘
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import cv2
import numpy as np

from .floormap import FloorMap
from .geometry import format_dimensions
from .heatmap import DensityField
from .mapview import floor_size, hotspot_label, render_map

BG = (14, 13, 12)
PANEL_BG = (20, 19, 17)
FG = (240, 240, 236)
MUTED = (146, 144, 138)
ACCENT = (70, 170, 250)
RULE = (52, 50, 47)


@dataclass
class DashboardStyle:
    width: int = 1920
    height: int = 1080
    camera_frac: float = 0.45
    title_h: int = 62
    stats_h: int = 86
    show_metrics: bool = True
    show_track_ids: bool = True
    show_zone_labels: bool = True
    show_roi_dimensions: bool = True
    show_scale_bar: bool = True
    show_heatmap_legend: bool = True
    show_hotspot: bool = True
    cumulative: bool = False
    heat_alpha: float = 0.78


def fit_into(img: np.ndarray, w: int, h: int, bg: tuple[int, int, int] = BG,
             out: np.ndarray | None = None) -> np.ndarray:
    """Letterbox into w x h. Aspect is never violated, so nothing is stretched.

    Pass ``out`` to reuse a panel across frames. Churning several megabytes of
    fresh buffers per frame costs more in page faults than the resize itself.
    """
    panel = np.full((h, w, 3), bg, np.uint8) if out is None else out
    panel[:] = bg
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return panel
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    if nw == iw and nh == ih:
        panel[y0:y0 + nh, x0:x0 + nw] = img      # exact fit, no resample needed
    else:
        # INTER_AREA is only worth its cost on a real downscale.
        interp = cv2.INTER_AREA if scale < 0.5 else cv2.INTER_LINEAR
        panel[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return panel


def format_dwell(seconds: float) -> str:
    s = int(max(seconds, 0))
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


class Dashboard:
    """Reusable composer. Buffers are allocated once, not per frame."""

    def __init__(self, fmap: FloorMap, style: DashboardStyle | None = None):
        self.fmap = fmap
        self.style = style or DashboardStyle()
        s = self.style
        self.cam_w = int(round(s.width * s.camera_frac))
        self.map_w = s.width - self.cam_w
        self.body_h = s.height - s.title_h - (s.stats_h if s.show_metrics else 0)
        self._frame = np.full((s.height, s.width, 3), BG, np.uint8)
        # Constant chrome is built once, not redrawn every frame.
        self._title_cache: np.ndarray | None = None
        self._cam_panel = fit_into(np.zeros((720, 1280, 3), np.uint8), self.cam_w, self.body_h)
        # Reused every frame: reallocating these costs more in page faults than
        # the resampling they exist for.
        self._cam_buf = np.empty((self.body_h, self.cam_w, 3), np.uint8)
        self._map_buf = np.empty((self.body_h, self.map_w, 3), np.uint8)
        mw, mh = floor_size(self.fmap, (self.map_w, self.body_h))
        self._map_canvas = np.empty((mh, mw, 3), np.uint8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    # ---- chrome ---------------------------------------------------------

    def _title_bar(self, img: np.ndarray, subtitle: str) -> None:
        s = self.style
        if self._title_cache is None:
            strip = np.full((s.title_h, s.width, 3), (26, 24, 22), np.uint8)
            cv2.line(strip, (0, s.title_h - 1), (s.width, s.title_h - 1), RULE, 1,
                     cv2.LINE_AA)
            cv2.putText(strip, "SMARTSTORE360", (26, 40), self.font, 1.05, FG, 2,
                        cv2.LINE_AA)
            w = cv2.getTextSize("SMARTSTORE360", self.font, 1.05, 2)[0][0]
            cv2.rectangle(strip, (26 + w + 18, 16), (30 + w + 18, 42), ACCENT, -1)
            cv2.putText(strip, subtitle, (46 + w, 36), self.font, 0.62, MUTED, 1,
                        cv2.LINE_AA)
            stamp = f"{s.width}x{s.height}"
            tw = cv2.getTextSize(stamp, self.font, 0.5, 1)[0][0]
            cv2.putText(strip, stamp, (s.width - tw - 26, 36), self.font, 0.5, MUTED,
                        1, cv2.LINE_AA)
            self._title_cache = strip
        img[0:s.title_h, :] = self._title_cache
        return

    def _stats_bar(self, img: np.ndarray, stats: dict) -> None:
        s = self.style
        y0 = s.height - s.stats_h
        img[y0:, :] = (24, 22, 21)
        cv2.line(img, (0, y0), (s.width, y0), RULE, 1, cv2.LINE_AA)

        hot = stats.get("hotspot")
        cells = [
            ("PEOPLE", str(stats.get("people", 0)), ACCENT),
            ("FOOTFALL", str(stats.get("footfall", 0)), FG),
            ("QUEUE", str(stats.get("queue", 0)), FG),
            ("AVG DWELL", format_dwell(stats.get("dwell", 0.0)), FG),
            ("HOTSPOT", hotspot_label(self.fmap, hot), (90, 140, 255)),
            ("FPS", f"{stats.get('fps', 0.0):.0f}", FG),
        ]
        n = len(cells)
        pad = 26
        slot = (s.width - pad * 2) // n
        for i, (label, value, color) in enumerate(cells):
            x = pad + i * slot
            cv2.putText(img, label, (x, y0 + 30), self.font, 0.46, MUTED, 1, cv2.LINE_AA)
            scale = 0.86 if len(value) < 8 else 0.62
            cv2.putText(img, value, (x, y0 + 66), self.font, scale, color, 2, cv2.LINE_AA)
            if i:
                cv2.line(img, (x - 14, y0 + 18), (x - 14, y0 + s.stats_h - 18), RULE, 1,
                         cv2.LINE_AA)

    # ---- composition ----------------------------------------------------

    def render(self, camera: np.ndarray, field: DensityField | None, points, labels,
               trails, occupancy, hot, stats) -> np.ndarray:
        s = self.style
        img = self._frame
        img[:] = BG
        self._title_bar(img, "LIVE STORE INTELLIGENCE")

        cam = fit_into(camera, self.cam_w, self.body_h, PANEL_BG, self._cam_buf)
        cv2.putText(cam, "SOURCE / CCTV", (12, 26), self.font, 0.55, (210, 210, 205), 1,
                    cv2.LINE_AA)
        cv2.rectangle(cam, (0, 0), (self.cam_w - 1, self.body_h - 1), RULE, 1, cv2.LINE_AA)

        mp = render_map(
            self.fmap, field, points, labels, trails, occupancy, hot,
            size=(self.map_w, self.body_h), cumulative=s.cumulative,
            heat_alpha=s.heat_alpha, show_zone_labels=s.show_zone_labels,
            show_dimensions=s.show_roi_dimensions, show_occupancy=s.show_roi_dimensions,
            show_track_ids=s.show_track_ids, show_trails=s.show_track_ids,
            show_hotspot=s.show_hotspot, show_scale_bar=s.show_scale_bar,
            show_legend=s.show_heatmap_legend, canvas=self._map_canvas,
        )
        mp = fit_into(mp, self.map_w, self.body_h, PANEL_BG, self._map_buf)
        cv2.rectangle(mp, (0, 0), (self.map_w - 1, self.body_h - 1), RULE, 1, cv2.LINE_AA)

        img[s.title_h:s.title_h + self.body_h, 0:self.cam_w] = cam
        img[s.title_h:s.title_h + self.body_h, self.cam_w:s.width] = mp
        if s.show_metrics:
            self._stats_bar(img, stats)
        return img


class TrailBuffer:
    """Short per-track position history for the motion-trail overlay."""

    def __init__(self, maxlen: int = 14):
        self.maxlen = maxlen
        self._h: dict[int, collections.deque] = {}

    def update(self, ids, pts: np.ndarray) -> dict[int, list]:
        if ids is None or len(pts) == 0:
            return {t: list(v) for t, v in self._h.items() if v}
        for tid, p in zip((int(t) for t in ids), pts):
            dq = self._h.get(tid)
            if dq is None:
                dq = self._h[tid] = collections.deque(maxlen=self.maxlen)
            dq.append((float(p[0]), float(p[1])))
        keep = {int(t) for t in ids}
        for tid in [t for t in self._h if t not in keep]:
            del self._h[tid]
        return {t: list(v) for t, v in self._h.items() if len(v) > 1}


def roi_report(fmap: FloorMap, occupancy: list[int]) -> list[dict]:
    """Per-named-zone real-world geometry plus live occupancy and density.

    The full record is for the JSON/JSONL outputs; the map draws a compact
    version of the same numbers.
    """
    out = []
    for i, zone in enumerate(fmap.zones):
        if zone.kind == "grid":
            continue
        m = dict(zone.measures)
        count = occupancy[i] if i < len(occupancy) else 0
        m["occupancy"] = count
        m["density"] = round(count / max(m["area_m2"], 0.25), 3)
        m["label"] = zone.label or zone.id
        m["dimensions"] = format_dimensions(m)
        out.append(m)
    return out
