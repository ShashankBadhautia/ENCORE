"""Configuration loading with defaults.

One YAML file, no validation framework. Every value has a default so a missing
key never crashes the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

AI_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = AI_ROOT / "config" / "config.yaml"

DEFAULTS: dict[str, Any] = {
    "camera": {
        "source": "0",
        "width": 1280,
        "height": 720,
        "fps": 30,
    },
    "model": {
        "weights": "yolo26s.pt",
        "imgsz": 640,
        "conf": 0.35,
        "tracker": "bytetrack.yaml",
        "engine": "",          # derived from weights when empty
        "half": True,
        "device": "",          # "", "0", "cpu"
    },
    "map": {
        "reference_points": None,   # [[x, y] x 4], clockwise; None -> click them
        "roi_polygon": None,        # [[x, y] x 4] camera px; None -> map everyone
        "width_m": 12.0,
        "height_m": 9.0,
        "grid_m": 1.0,
        "boundary_margin_m": 0.15,
        "zones": [],
    },
    "layout": {
        "calib_seconds": 20.0,
        "max_frames": 90,
        "mask_dilate_px": 24,
        "structure_gamma": 0.85,
    },
    # Continuous geospatial density field. See heatmap.DensityField.
    "heatmap": {
        "enabled": True,
        # Recent-movement window. `decay: null` derives per-second retention
        # from this as a half-life, which behaves the same at any frame rate.
        # Set `decay` explicitly to override (it is per-second retention).
        "window_seconds": 20.0,
        "decay": None,
        "sigma_m": 0.5,          # spatial smoothing radius, metres
        "resolution_px_per_m": 40.0,   # density raster sampling
        "max_kernel_px": 48,     # caps the per-person splat cost
        "cumulative": False,     # true = all-time instead of the recent window
    },
    "visualization": {
        "dashboard": True,       # write the side-by-side presentation stream
        "output_width": 1920,
        "output_height": 1080,
        "camera_frac": 0.45,     # 45% video / 55% map
        "show_track_ids": True,
        "show_zone_labels": True,
        "show_roi_dimensions": True,
        "show_scale_bar": True,
        "show_heatmap_legend": True,
        "show_metrics": True,
        "heat_alpha": 0.78,
    },
    "thresholds": {
        "density_high": 0.6,
    },
    "output": {
        "dir": "output",
        "layout": "output/layout.json",
        "structure": "output/structure.png",
        "metrics_csv": "output/metrics.csv",
        "metrics_jsonl": "output/metrics.jsonl",
        # Videos are written by default, so a bare `--source clip.mp4` gives
        # you both outputs. Set write_video: false for live-only runs.
        "write_video": True,
        "heatmap_video": "output/heatmap.mp4",
        "heatmap_px_per_m": 32.0,
        "accumulate": True,
        "annotated_video": "output/annotated.mp4",
        "dashboard_video": "output/smartstore360_dashboard.mp4",
        "roi_geometry_csv": "output/roi_geometry.csv",
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Read-only view over the merged configuration."""

    def __init__(self, data: dict, path: Path | None = None):
        self._d = _merge(DEFAULTS, data)
        self.path = path

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            return cls({}, None)
        with cfg_path.open() as fh:
            return cls(yaml.safe_load(fh) or {}, cfg_path)

    def section(self, name: str) -> dict:
        return self._d.get(name, {})

    def resolve(self, key: str) -> Path:
        """Resolve an output path relative to the AI package root."""
        p = Path(self.section("output").get(key, ""))
        return p if p.is_absolute() else AI_ROOT / p

    def __getitem__(self, name: str) -> dict:
        return self.section(name)

    @property
    def camera(self) -> dict:
        return self.section("camera")

    @property
    def model(self) -> dict:
        return self.section("model")

    @property
    def map(self) -> dict:
        return self.section("map")

    @property
    def layout(self) -> dict:
        return self.section("layout")

    @property
    def heatmap(self) -> dict:
        return self.section("heatmap")

    @property
    def visualization(self) -> dict:
        return self.section("visualization")

    @property
    def thresholds(self) -> dict:
        return self.section("thresholds")

    @property
    def output(self) -> dict:
        return self.section("output")
