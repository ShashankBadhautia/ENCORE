"""Stage B — live retail analytics on the calibrated 2D store map.

YOLO26 detects people, ByteTrack gives them identity, the homography puts their
foot-point on the floor, and polygon lookup turns that into metrics.

    smartstore-scan --source 0
    smartstore-scan --source rtsp://... --map-out AI/output/map.mp4
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import detector
from .config import Config
from .dashboard import Dashboard, DashboardStyle, TrailBuffer, roi_report
from .floormap import FloorMap
from .geometry import format_dimensions, polygon_measures
from .heatmap import DensityField, retention_for_window
from .layout import open_capture

CSV_FIELDS = [
    "t", "fps", "footfall_in", "footfall_out", "people",
    "congested_cells", "dwell_mean_s", "queue_max",
    # Spatial analytics, derived from the continuous density field.
    "hotspot_x_m", "hotspot_y_m", "hotspot_density", "hotspot_zone",
]


class VideoOut:
    """Lazily-opened video writer. The first frame fixes the size and fps."""

    def __init__(self, path: str, fps: float = 10.0):
        self.path = path
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")
PANEL_W, PANEL_H = 640, 360
PREVIEW_W, PREVIEW_H = 1280, 720


def fit_panel(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox an image into a w x h panel, preserving its aspect ratio.

    A plain cv2.resize to a fixed panel distorts: a 12x9 m floor shown in a
    640x360 box would render 25% too short, which misreads as a bad
    calibration. Side bars are padded instead.
    """
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    panel = np.full((h, w, 3), 18, dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    panel[y0 : y0 + nh, x0 : x0 + nw] = resized
    return panel


def clip_stem(source: str) -> str:
    """Filename stem of a file source, or "" for a live stream/camera index.

    Used to keep each test clip's videos separate: input/shop1.mp4 produces
    output/heatmap_shop1.mp4 and output/annotated_shop1.mp4.
    """
    s = str(source)
    if s.isdigit() or s.lower().startswith(LIVE_PREFIXES):
        return ""
    return Path(s).stem


def video_output(flag: str | None, cfg: Config, key: str, stem: str,
                 template: str | None = None) -> str | None:
    """Resolve a video output path: CLI flag wins, then config, then off.

    `template` names the file around the clip stem when the default ordering
    would read badly, e.g. smartstore360_shop1_dashboard.mp4.
    """
    if flag:
        return flag
    if not cfg.output.get("write_video", True):
        return None
    path = cfg.resolve(key)
    if not stem:
        return str(path)
    name = template.format(stem=stem) if template else f"{path.stem}_{stem}"
    return str(path.with_name(f"{name}{path.suffix}"))


class RoiGeometryWriter:
    """Static per-ROI real-world geometry, written once per run.

    Geometry does not change during a run, so this is written at startup. Live
    occupancy and density for the same ROIs go to metrics.jsonl every second.
    """

    FIELDS = [
        "id", "label", "kind", "area_m2", "width_m", "height_m", "angle_deg",
        "perimeter_m", "centroid_x_m", "centroid_y_m", "dimensions", "vertices",
    ]

    def __init__(self, path: Path, rows: list[dict]):
        self.rows = rows
        if not path:
            return
        try:
            with path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self.FIELDS, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        except OSError as exc:
            print(f"[scan] could not write ROI geometry: {exc}")


class Footfall:
    """Entry/exit counting, once per track ID per direction.

    If the map declares `kind: line` zones, crossings of those lines are counted
    (door-accurate). Otherwise the floor boundary is used.
    """

    def __init__(self, fmap: FloorMap):
        self.fmap = fmap
        self.lines = [z for z in fmap.zones if z.kind == "line"]
        self.states: dict = {}

    def _boundary_state(self, pts: np.ndarray) -> np.ndarray:
        f, m = self.fmap, self.fmap.boundary_margin_m
        return (
            (pts[:, 0] < -m).astype(np.int8)
            + (pts[:, 0] > f.width_m + m).astype(np.int8)
            + (pts[:, 1] < -m).astype(np.int8)
            + (pts[:, 1] > f.height_m + m).astype(np.int8)
        )

    def _update_lines(self, ids: np.ndarray, pts: np.ndarray) -> tuple[int, int]:
        """A `line` zone is a threshold strip: crossing into it is an entry,
        crossing out is an exit. Counting membership transitions (rather than
        the side of a chosen polygon edge) keeps this correct for any winding
        order of the configured strip."""
        entered = left = 0
        for zone in self.lines:
            on_line = zone.contains(pts)
            for i, tid in enumerate(ids):
                key = (int(tid), zone.id)
                current = 1 if on_line[i] else 0
                previous = self.states.get(key)
                if previous is None:
                    self.states[key] = current
                elif current != previous:
                    entered += current == 1
                    left += current == 0
                    self.states[key] = current
        return entered, left

    def _update_boundary(self, ids: np.ndarray, pts: np.ndarray) -> tuple[int, int]:
        state = self._boundary_state(pts)
        entered = left = 0
        for i, tid in enumerate(ids):
            tid, current = int(tid), int(state[i])
            previous = self.states.get(tid)
            if previous is None:
                self.states[tid] = current
            elif current != previous:
                entered += current == 0
                left += current != 0
                self.states[tid] = current
        return entered, left

    def update(self, ids: np.ndarray, pts: np.ndarray) -> tuple[int, int]:
        if len(ids) == 0:
            return 0, 0
        if self.lines:
            return self._update_lines(ids, pts)
        return self._update_boundary(ids, pts)

    def prune(self, ids: np.ndarray) -> None:
        """Drop state for tracks that are no longer in frame."""
        active = {int(t) for t in ids}
        for key in [k for k in self.states if (k[0] if isinstance(k, tuple) else k) not in active]:
            del self.states[key]


class Dwell:
    """Mean time-on-floor across completed tracks."""

    def __init__(self, maxlen: int = 200):
        self.first: dict[int, float] = {}
        self.last: dict[int, float] = {}
        self.completed: collections.deque = collections.deque(maxlen=maxlen)

    def update(self, ids: np.ndarray, now: float) -> float:
        current = {int(t) for t in ids}
        for tid in current:
            self.first.setdefault(tid, now)
            self.last[tid] = now
        for tid in [t for t in self.first if t not in current]:
            span = self.last[tid] - self.first.pop(tid)
            self.last.pop(tid, None)
            if 0 < span < 7200:
                self.completed.append(span)
        return self.mean

    @property
    def mean(self) -> float:
        return float(np.mean(self.completed)) if self.completed else 0.0


class MetricsWriter:
    def __init__(self, csv_path: Path, jsonl_path: Path):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl = jsonl_path.open("w")
        self.csv_fh = csv_path.open("w", newline="")
        self.csv = csv.DictWriter(self.csv_fh, fieldnames=CSV_FIELDS)
        self.csv.writeheader()

    def write(self, record: dict) -> None:
        self.jsonl.write(json.dumps(record) + "\n")
        self.jsonl.flush()
        self.csv.writerow({k: record.get(k) for k in CSV_FIELDS})
        self.csv_fh.flush()

    def close(self) -> None:
        self.jsonl.close()
        self.csv_fh.close()


def person_feet(results) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Track foot-points (bottom-centre of each box) in pixels, plus ids/confs.

    `ids` is None when the tracker confirmed no identities for this frame. That
    is reported rather than faked: occupancy and density only need positions,
    but footfall and dwell are per-identity, so inventing shared IDs would
    silently collapse several shoppers into one.
    """
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, None, empty
    xyxy = boxes.xyxy.cpu().numpy()
    feet = np.stack(
        [(xyxy[:, 0] + xyxy[:, 2]) / 2.0, np.clip(xyxy[:, 3], 0, results.orig_shape[0] - 1)],
        axis=1,
    ).astype(np.float32)
    ids = None if boxes.id is None else boxes.id.cpu().numpy().reshape(-1)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    return feet, ids, confs


def draw_camera(frame, xyxy, ids, confs, zone_names, fps: float,
                roi: list | None = None, outside: int = 0) -> np.ndarray:
    if roi and len(roi) >= 3:
        pts = np.array(roi, np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (255, 200, 0), 2, cv2.LINE_AA)
        x0 = int(min(p[0] for p in roi))
        y0 = int(min(p[1] for p in roi))
        cv2.putText(frame, "ROI", (x0, max(18, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 255, 120), 2)
        tag = f"#{int(ids[i])} {confs[i]:.2f}"
        if zone_names[i]:
            tag += f" {zone_names[i]}"
        cv2.putText(frame, tag, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 120), 1, cv2.LINE_AA)
    status = f"{fps:.1f} fps"
    if outside:
        status += f"  {outside} outside ROI"
    cv2.putText(frame, status, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def build_record(fmap: FloorMap, occ: list[int], people: int, total_in: int,
                 total_out: int, dwell_mean: float, fps: float, density_high: float,
                 field: DensityField | None = None) -> dict:
    named, grid, queue_max = [], [], 0
    for zone, count in zip(fmap.zones, occ):
        density = count / max(zone.area, 0.25)
        if zone.kind == "queue":
            queue_max = max(queue_max, count)
        if zone.kind != "grid":
            # Geometry travels with the occupancy so a consumer never has to
            # re-derive it: area_m2 / width_m / height_m are all included.
            entry = {"id": zone.id, "kind": zone.kind, "occupancy": count,
                     "density": round(density, 3), **zone.measures}
            entry["label"] = zone.label or zone.id
            entry["dimensions"] = format_dimensions(zone.measures)
            if zone.kind == "queue":
                entry["queue"] = count
            named.append(entry)
        elif count > 0:
            grid.append({"id": zone.id, "occupancy": count, "density": round(density, 3)})
    grid.sort(key=lambda cell: cell["occupancy"], reverse=True)
    congested = sum(1 for cell in grid if cell["density"] > density_high)

    hot = field.hotspot() if field is not None and field.enabled else None
    hot_zone = ""
    if hot:
        p = np.array([[hot["x_m"], hot["y_m"]]], dtype=np.float32)
        for zone in fmap.zones:
            if zone.kind != "grid" and zone.contains(p)[0]:
                hot_zone = zone.label or zone.id
                break

    rec = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fps": round(fps, 1),
        "footfall_in": total_in,
        "footfall_out": total_out,
        "people": people,
        "dwell_mean_s": round(dwell_mean, 1),
        "congested": congested > 0,
        "congested_cells": congested,
        "queue_max": queue_max,
        "hotspot_x_m": hot["x_m"] if hot else "",
        "hotspot_y_m": hot["y_m"] if hot else "",
        "hotspot_density": hot["density"] if hot else "",
        "hotspot_zone": hot_zone,
        "zones": named,
        "top_cells": grid[:3],
    }
    if field is not None and field.enabled:
        rec["hotspot"] = hot
        rec["peak_density"] = round(field.peak(), 3)
        rec["person_seconds"] = round(field.person_seconds(cumulative=True), 1)
    return rec


def run(cfg: Config, args) -> int:
    layout_path = Path(args.layout) if args.layout else cfg.resolve("layout")
    fmap = FloorMap.load(layout_path)
    print(f"[scan] map {fmap.width_m}m x {fmap.height_m}m, {len(fmap.zones)} zones")

    model, what = detector.load(cfg)
    print(f"[scan] model: {what} | tracker: {cfg.model['tracker']}")

    cap = open_capture(cfg, args.source)
    writer = MetricsWriter(cfg.resolve("metrics_csv"), cfg.resolve("metrics_jsonl"))
    footfall, dwell = Footfall(fmap), Dwell()

    # Written videos replay at the SOURCE rate, not a fixed default: at 10 fps
    # a 25 fps clip came out 2.5x slower than it really was.
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (0.1 < src_fps < 1000.0):
        src_fps = 25.0
        print("[scan] source reported no usable fps, assuming 25.0")

    stem = clip_stem(args.source if args.source is not None else cfg.camera["source"])
    # --no-heatmap suppresses the legacy grid heatmap video. The continuous
    # density field is a separate concern, controlled by heatmap.enabled, since
    # the dashboard renders it and should not go blank just because an extra
    # video file was declined.
    map_path = None if args.no_heatmap else video_output(args.map_out, cfg,
                                                          "heatmap_video", stem)
    cam_path = video_output(args.camera_out, cfg, "annotated_video", stem)
    dash_path = None
    if not args.no_dashboard and cfg.visualization.get("dashboard", True):
        dash_path = video_output(args.dashboard_out, cfg, "dashboard_video", stem,
                                 template="smartstore360_{stem}_dashboard")

    src_out = VideoOut(map_path, src_fps) if map_path else None
    cam_out = VideoOut(cam_path, src_fps) if cam_path else None
    dash_out = VideoOut(dash_path, src_fps) if dash_path else None
    map_out, cam_out_v = src_out, cam_out
    for label, path in (("dashboard", dash_path), ("heatmap", map_path),
                        ("annotated", cam_path)):
        if path:
            print(f"[scan] {label:9} -> {path} @ {src_fps:.2f} fps")

    imgsz, conf = int(cfg.model["imgsz"]), float(cfg.model["conf"])
    track_kwargs = {"device": cfg.model["device"]} if cfg.model.get("device") else {}
    density_high = float(cfg.thresholds["density_high"])

    # ---- continuous geospatial density field ----------------------------
    hc = cfg.heatmap
    field_on = bool(hc.get("enabled", True))
    dt = 1.0 / src_fps
    decay = hc.get("decay")
    if decay in (None, "", 0):
        # Half-life == window_seconds, so the window means the same thing at any
        # frame rate instead of depending on fps.
        decay = retention_for_window(float(hc["window_seconds"]), dt)
    field = DensityField(
        width_m=fmap.width_m, height_m=fmap.height_m,
        resolution=float(hc["resolution_px_per_m"]), sigma_m=float(hc["sigma_m"]),
        decay=float(decay), max_kernel_px=int(hc["max_kernel_px"]),
        enabled=field_on,
        boundary=fmap.floor_polygon,
    )
    if field.enabled:
        half_life = (np.log(0.5) / np.log(field.decay)) if 0 < field.decay < 1 else float("inf")
        print(f"[scan] density field {field.W}x{field.H} @ {field.resolution:g}px/m, "
              f"sigma {field.sigma_m:g}m, half-life {half_life:.0f}s")

    # ---- ROI geometry ---------------------------------------------------
    roi_rows = roi_report(fmap, [])
    RoiGeometryWriter(cfg.resolve("roi_geometry_csv"), roi_rows)
    for r in roi_rows:
        print(f"[scan] ROI {r['label']:10} {r['dimensions']}")
    if fmap.roi_polygon:
        fp = fmap.to_floor(np.asarray(fmap.roi_polygon, dtype=np.float32))
        if len(fp) >= 3:
            m = polygon_measures(fp, "watch_roi", "roi")
            print(f"[scan] watch ROI on the floor: {format_dimensions(m)}")

    # ---- presentation ---------------------------------------------------
    vc = cfg.visualization
    # Preview and saved output are independent: a frame may need rendering to be
    # written to the dashboard video without ever being shown, and a preview
    # panel may need showing without being written anywhere.
    mode = args.preview            # None | both | dashboard | map | camera
    show_dash = mode in ("both", "dashboard")
    show_map = mode in ("both", "map")
    show_cam = mode in ("both", "camera")

    dash = None
    if dash_out is not None or show_dash:
        dash = Dashboard(fmap, DashboardStyle(
            width=int(vc["output_width"]), height=int(vc["output_height"]),
            camera_frac=float(vc["camera_frac"]),
            show_metrics=bool(vc["show_metrics"]),
            show_track_ids=bool(vc["show_track_ids"]),
            show_zone_labels=bool(vc["show_zone_labels"]),
            show_roi_dimensions=bool(vc["show_roi_dimensions"]),
            show_scale_bar=bool(vc["show_scale_bar"]),
            show_heatmap_legend=bool(vc["show_heatmap_legend"]),
            cumulative=bool(hc["cumulative"]),
            heat_alpha=float(vc["heat_alpha"]),
        ))

    need_dash = dash is not None
    need_map = map_out is not None or show_map
    if mode:
        cv2.namedWindow("SmartStore360", cv2.WINDOW_NORMAL)
        if mode in ("map", "camera"):
            cv2.resizeWindow("SmartStore360", PREVIEW_W, PREVIEW_H)
        else:
            cv2.resizeWindow("SmartStore360",
                             int(vc["output_width"]), int(vc["output_height"]))
        print(f"[scan] preview: {mode} (q quits)")
    px_per_m = float(cfg.output.get("heatmap_px_per_m", 32.0))
    if show_map and not show_cam:
        px_per_m = min(PREVIEW_W / fmap.width_m, PREVIEW_H / fmap.height_m)
    panel = None

    fps_avg, frames, total_in, total_out = 30.0, 0, 0, 0
    dwell_mean = 0.0
    # Person-seconds per grid cell, for the legacy cell heatmap video and the
    # existing congestion metrics. The map itself uses the density field above.
    accum = np.zeros(len(fmap.zones), dtype=np.float64)
    show_accum = bool(cfg.output.get("accumulate", True))
    trails = TrailBuffer()
    hot: dict | None = None
    queue_max = 0
    last_emit, last_prune = 0.0, 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            results = model.track(
                frame, persist=True, tracker=cfg.model["tracker"], imgsz=imgsz,
                conf=conf, classes=[detector.PERSON_CLASS], verbose=False, **track_kwargs,
            )[0]
            feet, ids, confs = person_feet(results)
            xyxy = (results.boxes.xyxy.cpu().numpy() if results.boxes is not None
                    else np.zeros((0, 4), dtype=np.float32))

            # ROI filter runs on the camera-pixel foot point, before the
            # homography, so people outside the watched region are dropped
            # rather than folded into nonsense floor coordinates.
            n_seen = len(feet)
            if fmap.roi_polygon and n_seen:
                keep = fmap.in_roi(feet)
                if not keep.all():
                    feet, xyxy, confs = feet[keep], xyxy[keep], confs[keep]
                    if ids is not None:
                        ids = ids[keep]
            outside = n_seen - len(feet)

            # Chain preserved end to end: camera pixel -> foot point ->
            # homography -> real-world x/y metres -> density.
            floor_pts = fmap.to_floor(feet)
            on_floor = fmap.inside(floor_pts) if len(floor_pts) else np.zeros(0, bool)
            occ = fmap.occupancy(floor_pts)
            people = len(feet)
            if show_accum:
                for z, count in enumerate(occ):
                    if count:
                        accum[z] += count * dt
            cum_min = float(accum.sum()) / 60.0
            # Identities drive footfall and dwell, so those are only updated when
            # the tracker actually confirmed IDs this frame.
            labels = ids.astype(int) if ids is not None else np.arange(1, people + 1)

            if field.enabled:
                if len(floor_pts):
                    field.splat(floor_pts[on_floor], dt)
                field.advance(dt)
                hot = field.hotspot(cumulative=bool(hc["cumulative"]))

            if ids is not None:
                entered, left = footfall.update(ids, floor_pts)
                total_in += entered
                total_out += left
                dwell_mean = dwell.update(ids, time.time())

            frames += 1
            fps_avg = 0.9 * fps_avg + 0.1 / max(time.perf_counter() - t0, 1e-6)

            if cam_out_v is not None:
                zone_idx = fmap.zone_index_for(floor_pts)
                names = [fmap.zones[i].label if 0 <= i < len(fmap.zones) else "" for i in zone_idx]
                if cam_out_v is not None:
                    cam_out_v.write(draw_camera(frame.copy(), xyxy, labels, confs, names,
                                               fps_avg, fmap.roi_polygon, outside))

            if need_map:
                header = (f"in {total_in}  out {total_out}  people {people}  "
                          f"dwell {dwell_mean:.0f}s  {fps_avg:.1f} fps")
                if show_accum:
                    header += f"  cum {cum_min:.1f}m"
                if outside:
                    header += f"  (+{outside} out)"
                panel = fmap.render(occ, floor_pts, labels, header, px_per_m=px_per_m,
                                    accum=accum if show_accum else None)
                if map_out is not None:
                    map_out.write(panel)

            # ---- single synchronized presentation frame ------------------
            queue_max = max(
                (occ[i] for i, z in enumerate(fmap.zones) if z.kind == "queue"),
                default=0,
            )

            view = None
            if need_dash:
                view = dash.render(
                    frame, field, floor_pts, labels,
                    trails.update(ids, floor_pts) if ids is not None else None,
                    occ, hot,
                    {"people": people, "footfall": total_in, "queue": queue_max,
                     "dwell": dwell_mean, "fps": fps_avg, "hotspot": hot},
                )
                if dash_out is not None:
                    dash_out.write(view)

            if mode:
                if show_dash and view is not None:
                    shown = view
                elif show_map and show_cam:
                    shown = np.hstack([fit_panel(frame, PANEL_W, PANEL_H),
                                       fit_panel(panel, PANEL_W, PANEL_H)])
                elif show_map:
                    shown = fit_panel(panel, PREVIEW_W, PREVIEW_H)
                else:
                    shown = fit_panel(frame, PREVIEW_W, PREVIEW_H)
                cv2.imshow("SmartStore360", shown)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            now = time.time()
            if now - last_emit >= 1.0:
                last_emit = now
                record = build_record(fmap, occ, people, total_in, total_out,
                                      dwell_mean, fps_avg, density_high, field)
                writer.write(record)
                print(json.dumps({k: record[k] for k in CSV_FIELDS}), flush=True)
            if now - last_prune >= 5.0:
                last_prune = now
                if ids is not None:
                    footfall.prune(ids)
    except KeyboardInterrupt:
        print("\n[scan] stopped")
    finally:
        cap.release()
        writer.close()
        for out in (map_out, cam_out_v, dash_out):
            if out is not None:
                out.release()
        cv2.destroyAllWindows()

    print(f"[scan] {frames} frames, {total_in} in / {total_out} out")
    if field.enabled:
        print(f"[scan] density peak {field.peak():.2f} person-s/m2, "
              f"{field.person_seconds(cumulative=True) / 60:.1f} person-minutes on floor")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="smartstore-scan", description="Stage B: live analytics on the 2D store map"
    )
    ap.add_argument("--config", help="path to config.yaml")
    ap.add_argument("--source", help="webcam index, video path, or rtsp:// URL")
    ap.add_argument("--layout", help="path to layout.json")
    ap.add_argument("--map-out", help="write the 2D map stream to this video file")
    ap.add_argument("--camera-out", help="write the annotated camera stream to this video file")
    ap.add_argument(
        "--dashboard-out",
        help="write the combined side-by-side stream here "
             "(default output/smartstore360_<clip>_dashboard.mp4)",
    )
    ap.add_argument(
        "--no-dashboard", action="store_true",
        help="skip the combined stream (still writes the map and camera videos)",
    )
    ap.add_argument(
        "--no-heatmap", action="store_true",
        help="disable the continuous density field (grid metrics still work)",
    )
    ap.add_argument(
        "--preview", nargs="?", const="both", choices=("both", "dashboard", "map", "camera"),
        help="live window: the dashboard, a large map, or camera only (default: dashboard)",
    )
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    try:
        return run(cfg, args)
    except FileNotFoundError as exc:
        print(f"[scan] {exc}\n[scan] run 'smartstore-layout' first", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[scan] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
