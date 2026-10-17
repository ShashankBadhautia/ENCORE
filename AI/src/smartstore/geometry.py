"""Real-world measurements for polygons on the calibrated floor plane.

Everything in this module works in floor metres. The camera-pixel
`map.roi_polygon` is projected through the homography before it arrives here,
so no distance in the output is ever estimated from raw camera pixels.

Dimensions come from a minimum-area bounding rectangle, which is exact for the
rectangular zones people actually configure and a tight fit for irregular ones.
The values are labelled as estimated on screen: four clicked points cannot
deliver survey-grade accuracy.
"""

from __future__ import annotations

import cv2
import numpy as np


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of a polygon, in square units (m2 when input is metres)."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_perimeter(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return 0.0
    d = np.roll(p, -1, axis=0) - p
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros(2, dtype=np.float64)
    if len(p) < 3:
        return p.mean(axis=0)
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return p.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * area)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * area)
    return np.array([cx, cy], dtype=np.float64)


def min_area_rect(poly: np.ndarray) -> tuple[float, float, float]:
    """Width, height and rotation of the minimum-area bounding rectangle.

    The long side is always reported as `width`, so a portrait region reports
    its long edge first rather than flipping the label around.
    """
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    if len(p) < 2:
        return 0.0, 0.0, 0.0
    (_, _), (w, h), angle = cv2.minAreaRect(p)
    w, h = float(abs(w)), float(abs(h))
    if w < h:
        w, h = h, w
        angle += 90.0
    return w, h, float(angle % 180.0)


def polygon_measures(poly: np.ndarray, name: str = "", kind: str = "") -> dict:
    """Area / width / height / centroid for a polygon already in floor metres."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    w, h, angle = min_area_rect(p)
    centroid = polygon_centroid(p)
    return {
        "id": name,
        "kind": kind,
        "area_m2": round(polygon_area(p), 3),
        "width_m": round(w, 2),
        "height_m": round(h, 2),
        "angle_deg": round(angle, 1),
        "perimeter_m": round(polygon_perimeter(p), 2),
        "centroid_x_m": round(float(centroid[0]), 3),
        "centroid_y_m": round(float(centroid[1]), 3),
        "vertices": [[round(float(a), 4), round(float(b), 4)] for a, b in p],
    }


def format_dimensions(m: dict, digits: int = 1) -> str:
    """Compact '3.2 m x 5.8 m  (18.6 m2)' label for on-map text."""
    if not m or m.get("area_m2", 0) <= 0:
        return ""
    return "{:.{}f} m x {:.{}f} m  ({:.1f} m2)".format(
        m["width_m"], digits, m["height_m"], digits, m["area_m2"]
    )


def nice_scale_length(width_m: float, target_frac: float = 0.22) -> float:
    """Round scale-bar length (1/2/5 x 10^n) closest to `target_frac` of the map."""
    want = max(width_m * target_frac, 1e-6)
    for base in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        step = base
        while step * 10.0 <= want:
            step *= 10.0
        if abs(step - want) < 0.45 * want:
            return round(step, 3)
    return round(want, 3)
