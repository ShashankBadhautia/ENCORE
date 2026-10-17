# SmartStore360

**AI-powered retail intelligence built on infrastructure the store already has.**

A retail store already generates two valuable but disconnected streams of
information:

- **CCTV** — what is physically happening inside the store
- **Billing / POS** — what is happening in the business

SmartStore360 connects them through an **edge-first** intelligence layer on a
**Jetson Orin Nano** and turns them into operational and replenishment insight.

> **Observe → Understand → Predict → Act**
>
> **CCTV + POS → Edge AI → Analytics → Intelligence → Decision**

This repository currently implements the **CCTV path** — the two-stage vision
scanner that turns live store cameras into a 2D store map with live footfall,
occupancy, density, queue length and dwell analytics. The POS/billing path is
Phase 2 (see [Roadmap](#roadmap)).

---

## Architecture

```
EXISTING STORE INFRASTRUCTURE
        │
        ├──► NVR / VMS ──► security recording   (untouched, keeps running)
        │
        └──► live RTSP stream
                     │
                     ▼
            ┌──────────────────┐
            │   EDGE AI NODE   │   Jetson Orin Nano
            └────────┬─────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
  STAGE A (once)            STAGE B (always)
  layout.py                 scanner.py
        │                         │
  store calibration          YOLO26 + ByteTrack
  clip + 4 floor pts               │
        │                    foot-point per track
  temporal median                  │
  (people removed by YOLO26)        │
        │                           │
  homography ──────────────►  2D FLOOR MAP
        │                           │
   layout.json            footfall · occupancy · density
        │                  queue length · dwell · congestion
        └──────────► live metrics drawn on the map
```

**Existing CCTV is not replaced.** The RTSP stream is read in parallel with the
NVR. Raw video stays on-premises; only derived metadata leaves the store.

### Why two stages

Footfall means nothing without a map. So the store is calibrated **once**
(Stage A) to produce a rectified top-down 2D map, and the live scanner
(Stage B) renders every metric onto that map.

```
Store camera calibration clip
        ↓
YOLO26 removes people  +  temporal median isolates static structure
        ↓
4 floor reference points  →  homography  →  floor plane in metres
        ↓
layout.json  ──►  2D map  ──►  live analytics
```

Stage A does **not** ask YOLO26 to find shelves — COCO has no retail-fixture
class and that would be unreliable. YOLO26 does the one job it is genuinely
good at here: removing people from the calibration clip so a browsing customer
never gets baked into the store map.

---

## The scanner

YOLO26 handles detection. ByteTrack handles identity. Everything above that is
plain arithmetic in `floormap.py`.

```
YOLO26  ──►  who is visible            (yolo26s, person class only)
ByteTrack ─►  who is who, frame to frame (persistent track IDs)
foot-point ─► where they are on the floor (homography)
polygons ──►  which zone, crossed which boundary
```

### Metrics produced

| Metric | How it is derived |
|---|---|
| **Footfall in / out** | Track IDs crossing the floor boundary, counted once per ID |
| **Occupancy** | Live track foot-points inside each zone polygon |
| **Density** | Occupancy ÷ zone area (persons / m²) |
| **Queue length** | Occupancy of zones with `kind: queue` |
| **Dwell** | Per track ID, `last_seen − first_seen`, averaged over completed tracks |
| **Congestion** | Density above `thresholds.density_high` |

### What it deliberately does not do

SmartStore360 does **not** use vision to count stock on shelves. Inventory
prediction comes from the POS path. Keeping those separate is the point — see
[Roadmap](#roadmap).

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd AI
uv sync
```

### Stage A — build the 2D store map (once per camera)

```bash
# from a webcam, click the 4 floor corners when prompted
uv run smartstore-layout --source 0

# or from a store RTSP camera, non-interactive
uv run smartstore-layout \
  --source rtsp://user:pass@192.168.1.64/Streaming/Channels/101 \
  --points "520,610;1180,600;1210,690;480,700" \
  --calib-seconds 20
```

Writes `AI/output/layout.json` + `AI/output/structure.png`.

### Stage B — live analytics

```bash
uv run smartstore-scan --source 0
uv run smartstore-scan --source rtsp://user:pass@192.168.1.64/Streaming/Channels/101
```

```bash
# live window: the combined 1920x1080 dashboard (default), or one panel
uv run smartstore-scan --source 0 --preview            # dashboard
uv run smartstore-scan --source 0 --preview map        # full-size map only
uv run smartstore-scan --source 0 --preview camera     # camera only

# choose what gets written
uv run smartstore-scan --source 0 --dashboard-out AI/output/dash.mp4
uv run smartstore-scan --source 0 --no-dashboard --no-heatmap
```

`--preview` selects what is *displayed*; it never changes what is *saved*.
Every run writes `smartstore360_<clip>_dashboard.mp4`, `heatmap_<clip>.mp4`
and `annotated_<clip>.mp4` unless suppressed. `--no-heatmap` drops the legacy
grid heatmap video only — the continuous density field behind the dashboard is
controlled by `heatmap.enabled`.

Live records stream to stdout as JSON lines and append to
`AI/output/metrics.csv`.

---

## Configuration

`AI/config/config.yaml` — every CLI flag can also be set here.

```yaml
camera:
  source: "0"          # webcam index, video path, or rtsp:// URL
  width: 1280
  height: 720
  fps: 30

model:
  weights: "yolo26s.pt" # yolo26n / yolo26s / yolo26m / yolo26l / yolo26x
  imgsz: 640
  conf: 0.35
  tracker: "bytetrack.yaml"

map:
  reference_points: null   # [[x,y] x4] clockwise, or null to click them
  width_m: 12.0            # real width of the floor rectangle
  height_m: 9.0            # real depth of the floor rectangle
  grid_m: 1.0              # zone grid cell size
  boundary_margin_m: 0.15  # hysteresis for footfall crossings
  zones:                   # named zones overlaid on the grid
    - id: "entrance"
      kind: "line"
      polygon: [[5.4, 0.0], [6.6, 0.0], [6.6, 0.9], [5.4, 0.9]]
    - id: "checkout"
      kind: "queue"
      polygon: [[9.0, 5.0], [12.0, 5.0], [12.0, 9.0], [9.0, 9.0]]

thresholds:
  density_high: 0.6        # persons/m² before congestion is flagged

heatmap:                   # continuous density field behind the dashboard
  enabled: true
  resolution_px_per_m: 40   # field grid; the map is upscaled smoothly
  sigma_m: 0.5              # Gaussian bandwidth, in floor metres
  decay: null               # null => half-life == window_seconds
  window_seconds: 20
  max_kernel_px: 48         # cap on splat radius, for crowded frames
  cumulative: false         # true => show accumulated presence, not recent

visualization:
  dashboard: true
  output_width: 1920
  output_height: 1080
  camera_frac: 0.45         # left/right split
  show_metrics: true
  show_track_ids: true
  show_zone_labels: true
  show_roi_dimensions: true # real-world ROI size on the map
  show_scale_bar: true
  show_heatmap_legend: true
  show_hotspot: true
  heat_alpha: 0.78

output:
  dir: "output"
  metrics_csv: "output/metrics.csv"
  metrics_jsonl: "output/metrics.jsonl"
  layout: "output/layout.json"
  dashboard_video: "output/smartstore360_dashboard.mp4"
  roi_geometry_csv: "output/roi_geometry.csv"
```

`heatmap.decay` is the fraction of density retained per second. Leaving it
`null` makes `window_seconds` mean the same thing at any frame rate, instead of
depending on fps.

### Output schema

```json
{"t":"2026-09-27T18:04:12Z","fps":27.4,
 "footfall_in":41,"footfall_out":38,"people":23,
 "dwell_mean_s":143.2,"congested":false,
 "hotspot_x_m":6.2,"hotspot_y_m":7.1,"hotspot_density":1.42,
 "hotspot_zone":"checkout","peak_density":1.42,"person_seconds":31.8,
 "zones":[{"id":"checkout","kind":"queue","occupancy":5,"density":0.83,"queue":5,
           "width_m":4.0,"height_m":3.0,"area_m2":12.0}]}
```

`roi_geometry.csv` records the measured geometry of every configured ROI once,
at startup: `width_m`, `height_m`, `area_m2`, orientation, and vertices.

---

## Performance

Nothing is traded away for simplicity — the defaults are the fast ones.

| Decision | Effect |
|---|---|
| YOLO26 end-to-end head (NMS-free, default) | No NMS stage at all — faster *and* less config |
| `classes=[0]` | Person class only; 79 classes of post-processing dropped |
| `yolo26s`, FP16 TensorRT | Best accuracy/watt on Orin Nano 8GB. Engine is built once and cached |
| `imgsz=640` | YOLO26's native sweet spot on Orin |
| `cv2.VideoCapture` direct loop | Single pass, no frame-queue latency |
| Metrics buffered to 1 Hz | No per-frame I/O stall in the inference loop |
| Vectorised point-in-polygon (`numpy`) | Zone lookup is one array op, not a Python loop per person |
| Local Gaussian splats, not a full-grid KDE | Density update is O(people × kernel), not O(cells) |
| Cached tinted structure backdrop | The rectified floor plan is built once, not 25×/s |
| Gamma via 4096-entry LUT | Avoids `np.power` on every raster cell |
| `cv2.addWeighted` + masked `copyTo` for zones | SIMD blend instead of boolean fancy indexing |
| Reused panel buffers | No multi-megabyte allocation churn per frame |
| `INTER_AREA` only below 0.5× | Skips a filter that buys nothing on mild downscales |

Measured on this dev machine (CPU, no TensorRT — the Orin numbers will
differ, but the ratio is the point): YOLO26s + ByteTrack is ~41 ms/frame and
the entire visualisation path — density splat, smooth map render, 1920×1080
composition and mp4v encode — is ~18 ms. Inference is 69% of the per-frame
budget, so the map is no longer the bottleneck. The ceiling is ~17 fps.

Model size is a one-line change in `config.yaml`. Drop to `yolo26n` if you are
running four or more concurrent camera streams on a single Orin Nano.

---

## Jetson Orin Nano notes

1. Install JetPack, then PyTorch + TensorRT from NVIDIA's redistributor for
   your JetPack release — do **not** let PyPI pick them.
2. `uv` is awkward against a system Torch build. Prefer a system virtualenv and
   `uv pip install`:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   uv pip install --no-deps ultralytics opencv-python PyYAML
   uv pip install -r requirements.txt   # remaining pure-python deps
   ```
3. Confirm OpenCV sees CUDA:
   ```bash
   python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
   ```
   If that is `0`, the decoder is running on CPU and the loop is CPU-bound.
4. H.265 RTSP streams must be decoded by **GStreamer**, not OpenCV's FFmpeg
   backend, or you will get a 2 fps stream. Set `OPENCV_FFMPEG_CAPTURE_OPTIONS`
   or transcode the camera to H.264.

---

## Honest limitations

- **Stage A is per-camera and manual.** One calibration clip and four clicked
  points per camera view. Roughly 30 seconds, once, at install.
- **v1 map is floor bounds + zone grid.** Blocked-vs-walkable cell detection
  (which floor cells are shelf/counter) is not implemented yet. Structure is
  rectified and visible, not semantically classified. The legacy grid heatmap
  video still renders cells; the dashboard map does not.
- **Density is a decaying estimate, not a count.** `window_seconds` is a
  half-life, so a person who left 30 seconds ago still contributes a little.
  That is deliberate — it keeps the map readable — but it is not a census.
- **ROI dimensions are estimated from the calibration**, not surveyed. A
  homography fitted to four clicked points is good to a few centimetres over a
  12 m floor, which is fine for occupancy analytics and not for construction.
- **Footfall is a boundary-crossing estimate.** A person occluded at the door,
  or a camera that cannot see the full floor, will skew it. Zones declared with
  `kind: line` give you door-accurate counts.
- **One camera view at a time.** Multi-camera fusion needs a shared floor
  calibration and is not built.
- **Dwell is computed on track IDs**, so a heavy occlusion that drops an ID
  shortens the measured dwell for that track.

---

## Roadmap

```
[done]  Stage A — 2D store map from calibration clip
[done]  Stage B — footfall / occupancy / density / queue / dwell
[done]  Continuous Gaussian density field, floor-masked, with temporal decay
[done]  Synchronized 1920x1080 camera + map dashboard
[done]  Measured ROI geometry (area, dimensions, orientation) in outputs
[next]  Blocked-cell detection so congestion is computed per aisle, not per cell
[next]  POS / billing ingest  →  billing analytics  →  demand + stock prediction
[next]  Intelligence / decision engine — spatial + predictive + contextual fusion
[next]  Evaluation layer (significance, anomaly, opportunity)  →  alerts
[next]  Backend service + dashboard
```

---

## Repository layout

```
LICENSE
README.md
AI/                        edge AI node — the intelligence pipeline
  pyproject.toml
  config/config.yaml
  src/smartstore/
    config.py              configuration with defaults
    detector.py            YOLO26 load + TensorRT engine caching
    floormap.py            homography, zones, ROI filtering, 2D rendering
    geometry.py            real-world polygon area / perimeter / dimensions
    heatmap.py             continuous Gaussian density field + temporal decay
    mapview.py             smooth masked heat map, zones, trails, legend
    dashboard.py           synchronized camera + map compositor
    layout.py              Stage A — build the 2D store map
    scanner.py             Stage B — live analytics on the map
  models/                  weights + cached .engine   (gitignored)
  output/                  layout.json, metrics, video (gitignored)
```

`Backend/` and `Dashboard/` will sit alongside `AI/` when added.

---

## License

Project code: **MIT** — see [LICENSE](LICENSE).

**Dependency notice:** `ultralytics` (which provides YOLO26 and ByteTrack) is
**AGPL-3.0 / Enterprise**, not MIT. This repository is MIT and depends on
AGPL-3.0 software. The MIT terms cover *this project's own source only*; the
Ultralytics package carries its own licence, and a distributed deployment must
comply with AGPL-3.0 (or obtain an Ultralytics Enterprise licence). YOLO26
model weights inherit the same terms.
