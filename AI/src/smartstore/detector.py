"""YOLO26 loading, with a cached FP16 TensorRT engine.

The engine is expensive to build, so it is built once and reused. If the export
is not possible on this machine (no CUDA, no TensorRT), we fall back to the
PyTorch weights silently rather than failing the run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

PERSON_CLASS = 0


def is_cpu(device: str) -> bool:
    return str(device).strip().lower() == "cpu"


def _default_engine_name(weights: str) -> str:
    digest = hashlib.sha1(str(Path(weights).resolve()).encode()).hexdigest()[:8]
    return f"{Path(weights).stem}.{digest}.engine"


def detect_persons(model: Any, frame, imgsz: int, conf: float) -> list[tuple[int, int, int, int]]:
    """Return person boxes as (x1, y1, x2, y2) in pixels."""
    result = model.predict(
        frame,
        imgsz=imgsz,
        conf=conf,
        classes=[PERSON_CLASS],
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    return [tuple(int(round(v)) for v in row[:4]) for row in xyxy]


def load(cfg) -> tuple[Any, str]:
    """Load the detector. Returns (model, description of what was loaded)."""
    from ultralytics import YOLO

    m = cfg.model
    weights = m["weights"]

    if weights.endswith(".engine"):
        return YOLO(weights), weights

    # The derived engine name includes a digest of the weights path, so two
    # fine-tunes that both end in best.pt cannot silently share one engine.
    engine_path = Path(m.get("engine") or _default_engine_name(weights))
    if engine_path.exists():
        return YOLO(str(engine_path)), str(engine_path)

    model = YOLO(weights)
    half = bool(m["half"]) and not is_cpu(m.get("device", ""))
    try:
        exported = model.export(
            format="engine",
            imgsz=m["imgsz"],
            half=half,
            device=m.get("device") or None,
        )
        built = Path(exported) if isinstance(exported, (str, Path)) else engine_path
        if built.exists():
            return YOLO(str(built)), str(built)
    except Exception as exc:  # no TensorRT / unsupported platform
        print(f"[detector] TensorRT export unavailable ({exc}); using {weights}")

    if is_cpu(m.get("device", "")):
        model.to("cpu")
    return model, weights
