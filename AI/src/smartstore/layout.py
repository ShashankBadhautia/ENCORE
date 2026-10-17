"""Stage A — build the 2D store map from a calibration clip.

YOLO26's only job here: find the people in the clip so they can be erased from
the temporal median. What is left is the static structure of the store, which
we rectify onto the floor plane using four reference points.

    smartstore-layout --source 0
    smartstore-layout --source rtsp://... --points "x,y;x,y;x,y;x,y"
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from . import detector
from .config import Config
from .floormap import FloorMap


def open_capture(cfg: Config, source: str | None = None):
    cam = cfg.camera
    src = source if source is not None else str(cam["source"])
    if src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {src}")
    if cam.get("width"):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam["width"]))
    if cam.get("height"):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam["height"]))
    if cam.get("fps"):
        cap.set(cv2.CAP_PROP_FPS, int(cam["fps"]))
    return cap


def sample_frames(cap, seconds: float, max_frames: int) -> list[np.ndarray]:
    """Evenly sample frames across a window of the stream."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = max(int(fps * seconds), 2)
    grabbed: dict[int, np.ndarray] = {}
    idx = 0
    while idx < total:
        ok, frame = cap.read()
        if not ok:
            break
        grabbed[idx] = frame
        idx += 1
    if not grabbed:
        raise RuntimeError("no frames read from source")
    keys = sorted(grabbed)[:: max(1, len(grabbed) // max_frames)][:max_frames]
    return [grabbed[k] for k in keys]


def people_mask(model, frames: list[np.ndarray], imgsz: int, conf: float, dilate: int):
    """Union of padded person boxes across the clip, resized to the frame."""
    h, w = frames[0].shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    kernel = np.ones((dilate, dilate), np.uint8) if dilate > 1 else None
    for frame in frames:
        for x1, y1, x2, y2 in detector.detect_persons(model, frame, imgsz, conf):
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        if kernel is not None:
            mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def build_structure(model, frames: list[np.ndarray], cfg: Config) -> np.ndarray:
    """Temporal median with people removed, above the floor horizon zeroed out."""
    stack = np.stack(frames)
    median = np.median(stack, axis=0).astype(np.uint8)
    mask = people_mask(
        model,
        frames,
        int(cfg.model["imgsz"]),
        float(cfg.model["conf"]),
        int(cfg.layout["mask_dilate_px"]),
    )
    median[mask > 0] = 0
    return median


def pick_points(frame: np.ndarray, count: int = 4, label: str = "floor",
                hint: str = "click", optional: bool = False) -> list[list[int]]:
    """Interactive n-point picker. Returns points in click order.

    With `optional`, q/ESC skips and returns [] instead of aborting, so the ROI
    step never traps a user who does not need one.
    """
    points: list[list[int]] = []
    window = f"SmartStore360 - {label}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    def on_click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            points.append([int(x), int(y)])
            print(f"  point {len(points)}: ({x}, {y})")

    cv2.setMouseCallback(window, on_click)
    preview = frame.copy()

    def overlay():
        for i, (x, y) in enumerate(points):
            cv2.circle(preview, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(preview, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if count > 4:
            cv2.polylines(preview, [np.array(points, np.int32)], True, (0, 255, 255), 2)

    skip = " (optional, q to skip)" if optional else ""
    print(f"click {count} {label} points clockwise ({hint}){skip}; q/enter when done")
    while True:
        overlay()
        cv2.imshow(window, preview)
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13, 32) and len(points) == count:
            break
        if len(points) == count:
            break
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            if optional:
                return []
            raise SystemExit("aborted")
    cv2.destroyAllWindows()
    if len(points) != count:
        if optional:
            return []
        raise SystemExit(f"need exactly {count} reference points")
    return points


def parse_points(raw: str, count: int = 4) -> list[list[int]]:
    pts = [[int(v) for v in pair.split(",")] for pair in raw.split(";")]
    if len(pts) != count or any(len(p) != 2 for p in pts):
        raise SystemExit(f"--points must be '{';'.join(['x,y'] * count)}'")
    return pts


def build_layout(cfg: Config, source: str | None, points_arg: str | None,
                 calib_seconds: float | None,
                 roi_arg: str | None = None, want_roi: bool = False) -> FloorMap:
    cap = open_capture(cfg, source)
    try:
        print("[layout] sampling calibration clip...")
        frames = sample_frames(cap, calib_seconds or float(cfg.layout["calib_seconds"]),
                               int(cfg.layout["max_frames"]))
    finally:
        cap.release()
    print(f"[layout] {len(frames)} frames sampled")

    model, what = detector.load(cfg)
    print(f"[layout] model: {what}")

    print("[layout] removing people and taking the temporal median...")
    structure = build_structure(model, frames, cfg)

    first = frames[len(frames) // 2]
    if points_arg:
        points = parse_points(points_arg)
    elif cfg.map.get("reference_points"):
        points = [list(map(int, p)) for p in cfg.map["reference_points"]]
        print(f"[layout] reference points from config: {points}")
    else:
        points = pick_points(first)

    # ROI: which part of the camera view counts as "the store". Optional, and
    # stored in layout.json so the scanner needs no extra flag. Only prompted
    # when the floor corners were clicked too, so scripted runs (which pass
    # --points) never block on a window that may not exist.
    interactive = not points_arg and not cfg.map.get("reference_points")
    if roi_arg:
        roi = parse_points(roi_arg, count=4)
    elif cfg.map.get("roi_polygon"):
        roi = [[float(v) for v in p] for p in cfg.map["roi_polygon"]]
        print(f"[layout] ROI from config: {roi}")
    elif interactive or want_roi:
        roi = pick_points(
            first, count=4, label="ROI corners", optional=True,
            hint="the region to watch; press q to watch everything",
        )
    else:
        roi = []
    if roi:
        print(f"[layout] ROI: {[[int(x), int(y)] for x, y in roi]}")
    else:
        print("[layout] no ROI: every detected person is mapped")

    h, w = structure.shape[:2]
    if any(not (0 <= x < w and 0 <= y < h) for x, y in points):
        raise SystemExit(f"reference points outside the {w}x{h} frame")
    if roi and any(not (0 <= x < w and 0 <= y < h) for x, y in roi):
        raise SystemExit(f"ROI points outside the {w}x{h} frame")
    structure[: min(y for _, y in points) - 1, :] = 0

    out_struct = cfg.resolve("structure")
    out_struct.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_struct), structure)
    print(f"[layout] structure -> {out_struct}")

    fmap = FloorMap.from_reference_points(
        points,
        float(cfg.map["width_m"]),
        float(cfg.map["height_m"]),
        grid_m=float(cfg.map["grid_m"]),
        boundary_margin_m=float(cfg.map["boundary_margin_m"]),
        named_zones=cfg.map.get("zones") or [],
        structure_path=str(out_struct),
        structure_gamma=float(cfg.layout["structure_gamma"]),
        roi_polygon=roi or None,
        meta={
            "source": str(source if source is not None else cfg.camera["source"]),
            "frame_size": [w, h],
            "reference_points": points,
            "calib_frames": len(frames),
            "weights": cfg.model["weights"],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    out_layout = cfg.resolve("layout")
    fmap.save(out_layout)
    print(f"[layout] {len(fmap.zones)} zones, {fmap.width_m}m x {fmap.height_m}m")
    print(f"[layout] layout  -> {out_layout}")
    return fmap


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="smartstore-layout", description="Stage A: build the 2D store map"
    )
    ap.add_argument("--config", help="path to config.yaml")
    ap.add_argument("--source", help="webcam index, video path, or rtsp:// URL")
    ap.add_argument(
        "--points", help="four floor corners as 'x,y;x,y;x,y;x,y' (skips the click prompt)"
    )
    ap.add_argument(
        "--roi-points",
        help="ROI corners as 'x,y;x,y;x,y;x,y' - only people here are mapped",
    )
    ap.add_argument("--roi", action="store_true",
                    help="prompt for ROI corners even when --points is given")
    ap.add_argument("--calib-seconds", type=float, help="calibration clip length")
    ap.add_argument("--preview", action="store_true", help="show the rectified store map")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    try:
        fmap = build_layout(cfg, args.source, args.points, args.calib_seconds,
                            args.roi_points, args.roi)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[layout] failed: {exc}", file=sys.stderr)
        return 1

    if args.preview:
        canvas = fmap.render(header="calibrated store map - q to close")
        cv2.imshow("SmartStore360 - map", canvas)
        while cv2.waitKey(30) & 0xFF not in (27, ord("q")):
            pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
