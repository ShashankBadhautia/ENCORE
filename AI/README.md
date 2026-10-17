# SmartStore360 — Edge AI node

The intelligence pipeline. Runs on a Jetson Orin Nano inside the store.

| Module | Stage | What it does |
|---|---|---|
| `config.py` | — | Loads `config/config.yaml`, fills defaults |
| `detector.py` | — | Loads YOLO26, builds/caches the FP16 TensorRT engine |
| `floormap.py` | — | Homography, zone geometry, ROI filtering, 2D map rendering |
| `geometry.py` | — | Real-world polygon area/perimeter/dimensions |
| `heatmap.py` | — | Continuous Gaussian density field, temporal decay, floor mask, hotspot |
| `mapview.py` | — | Smooth masked heat map, zones, trails, hotspot, scale, legend |
| `dashboard.py` | — | Synchronized camera + map compositor |
| `layout.py` | A | Calibration clip → `layout.json` + rectified store map |
| `scanner.py` | B | Live stream → footfall, occupancy, density, queue, dwell, dashboard |

## Setup

```bash
uv sync
```

## Run

```bash
uv run smartstore-layout --source 0     # Stage A, once per camera
uv run smartstore-scan   --source 0     # Stage B, live
```

Stage B writes, by default:

| File | Contents |
|---|---|
| `smartstore360_<clip>_dashboard.mp4` | 1920×1080 camera + live map, frame-locked to the source |
| `heatmap_<clip>.mp4` | Legacy grid heatmap |
| `annotated_<clip>.mp4` | Camera with boxes, foot points, zone names, ROI outline |
| `metrics.csv` / `metrics.jsonl` | Per-second metrics including hotspot and peak density |
| `roi_geometry.csv` | Static real-world dimensions of every configured ROI |

## Preview

```bash
uv run smartstore-scan --source 0 --preview dashboard   # default: the combined view
uv run smartstore-scan --source 0 --preview map         # full-size map only
uv run smartstore-scan --source 0 --preview camera      # camera only
```

`--preview` chooses what is *displayed*; it does not change what is *saved*.
`--no-dashboard` and `--no-heatmap` suppress the respective video files. The
continuous density field that feeds the dashboard is controlled separately by
`heatmap.enabled` in the config.

All flags have equivalents in `config/config.yaml`. See the root
[README](../README.md) for the full configuration and output reference.
