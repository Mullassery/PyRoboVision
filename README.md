# PyRoboVision

**Tracking and trajectory prediction that runs on CPU — with a clean boundary to NVIDIA's Alpamayo stack.** Pure Python (NumPy + SciPy) multi-object tracking, occlusion handling, and motion forecasting for robotics.

[![CI](https://github.com/Mullassery/PyRoboVision/actions/workflows/tests.yml/badge.svg)](https://github.com/Mullassery/PyRoboVision/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/pyrobovision)](https://pypi.org/project/pyrobovision/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Alpamayo%20adapter-76B900?logo=nvidia&logoColor=white)](#nvidia-ecosystem)

> **Why it exists:** most robot stacks need identity persistence and short-horizon motion prediction long before they need another foundation model. PyRoboVision gives you a tested Kalman + Hungarian tracker, a trajectory predictor with uncertainty, and adapters for the models you already run — without dragging PyTorch into your perception core.

## 60-second quickstart

```bash
pip install pyrobovision
```

```python
import numpy as np
from pyrobovision import MOTTracker, Detection

tracker = MOTTracker(max_age=30, min_hits=3)
detections = [Detection(bbox=np.array([100, 100, 140, 180]), confidence=0.92)]

for track in tracker.update(detections):
    print(track.track_id, track.get_position(), track.get_velocity())
```

No detector is bundled — you bring bounding boxes from YOLO, SAM, MMDetection, or your own model. See the [Supervision adapter](#bring-your-own-detector-supervision-adapter).

## NVIDIA ecosystem

PyRoboVision is **NVIDIA-optional**: the tracker, predictor, depth utilities, and tests run on CPU with NumPy + SciPy. The NVIDIA touchpoint is an explicit model-output boundary, not a hidden dependency.

| Component | Status | What it actually does |
|---|---|---|
| Alpamayo trajectory + reasoning adapter | **Available** | Normalizes Alpamayo NumPy/tensor outputs into `(x, y, z)` waypoints, rotations, and per-sample reasoning. Does **not** import `torch` or `alpamayo_r1`, and does **not** run the model. |
| Alpamayo 1.5 / 2 contract fixtures | **Available** | CPU-only synthetic fixtures exercise the same 64-waypoint / 10 Hz contract in CI. They are not real model output. |
| Alpamayo model execution | **External** | Alpamayo 1 weights are research/non-commercial and inference needs an NVIDIA GPU; install and run upstream, then pass outputs to the adapter. |
| SANA-WM video generation / evaluation | **Not integrated** | No SANA-WM loader, generator, or benchmark ships. The tracker is model-agnostic over frames, but synthetic-video evaluation is not part of this release. |

**The honest headline:** this is the CPU perception layer around an NVIDIA model — not an NVIDIA-accelerated tracker. That boundary is intentional: your tracking tests stay fast and deterministic while Alpamayo runs wherever its license and GPU allow.

## At a glance

| Area | Status |
|---|---|
| `MOTTracker` / `Track` / `Detection` | **Stable** — Kalman + Hungarian, birth/confirm/occlude/death |
| Occlusion re-association | **Stable**, unit-tested |
| `TrajectoryPredictor` (CV / CA + uncertainty) | **Stable** |
| `ByteTrackTracker` two-stage association | **Available** |
| Supervision detector adapter | **Available** |
| Alpamayo adapter + synthetic fixtures | **Available** |
| MiDaS depth (`[depth]`) | **Optional**, real pretrained model |
| `DepthEstimator(model="heuristic")` | **Placeholder** — Sobel-edge heuristic, not learned depth |
| Detector / GPU inference pipeline | **Not included** |
| SANA-WM integration | **Not shipped** |

## What it is

PyRoboVision's real, tested core is a multi-object tracker: Kalman-filter state estimation plus Hungarian-algorithm association, with occlusion handling (tracks are predicted through detection gaps and re-associate on reacquisition) and constant-velocity / constant-acceleration trajectory prediction with uncertainty. Around it are supporting utilities: 3D perception (depth, 3D boxes, LiDAR, occupancy grids), rule-based behavior/intent classification, and minimal imitation-learning / behavior-cloning / safety-constraint building blocks. The core depends on NumPy and SciPy only; PyTorch is optional.

## Features

- **Multi-object tracking** — `MOTTracker`, `Track`, `Detection`; birth/confirm/occlude/death lifecycle; IoU association via `scipy.optimize.linear_sum_assignment`.
- **Occlusion handling** — every track is predicted each frame; a confirmed track survives up to `max_age` missed frames and re-associates to the same `track_id`.
- **Trajectory prediction** — `TrajectoryPredictor` with `"cv"` (constant velocity) or `"ca"` (constant acceleration) models and uncertainty.
- **ByteTrack-style association** — `ByteTrackTracker` keeps low-confidence detections for a second pass, improving continuity through short occlusions without AGPL code.
- **Depth estimation** — `DepthEstimator(model="heuristic")` (default, no extra deps) or `model="midas"` (real pretrained MiDaS_small via `torch.hub`, optional `[depth]` extra).
- **3D perception utilities** — `Box3DConverter` (2D bbox + depth → 3D), LiDAR point-cloud filtering/clustering, BEV occupancy grids, plus `OccupancyGrid.raycast()` / `line_of_sight()` for sensor line-of-sight checks.
- **Sensor fusion** — Kalman-based IMU/GPS fusion with covariance (`SensorFusionEngine`).
- **ONNX export** — thin wrapper around `torch.onnx.export` (optional `[onnx]` extra).
- **Learning helpers** — imitation learning, behavior cloning, `SafetyValidator` constraint correction.

## Who it's for / Use cases

- **Adding tracking on top of your own detector** — feed bounding boxes from YOLO/SAM/anything into `MOTTracker`; it handles identity persistence through occlusion.
- **Forecasting from tracked positions** — constant-velocity/acceleration prediction with uncertainty for downstream collision-avoidance logic.
- **Consuming Alpamayo outputs on CPU** — parse ego trajectories and reasoning without importing the model stack.
- **IMU/GPS fusion** — a fused position estimate with covariance without hand-rolling a Kalman filter.
- **Prototyping 3D perception** — depth maps, 3D boxes, and occupancy grids from 2D detections and a calibrated camera.

## What it is not

- **No object detector.** You bring your own detections (bounding boxes + confidence); none is bundled or wrapped.
- **No GPU-accelerated inference pipeline.** The tracking/prediction core is CPU NumPy/SciPy.
- **No foundation-model integration** (CLIP/SAM/Grounding DINO). An earlier version shipped a module that returned hardcoded fake results; it was removed, not fixed.
- **MiDaS output is relative, not metric** — correctly ordered near/far depth, not meters, unless you calibrate against a known reference.
- **Not a certified or safety-verified autonomous-driving stack.** `SafetyValidator` is a basic constraint checker.
- **The default `heuristic` depth backend is a Sobel-edge placeholder, not a learned depth model.**
- **Alpamayo is not executed in-process.** The adapter consumes outputs; model weights, licensing, and GPU requirements stay upstream.

## ByteTrack-style two-stage association

`ByteTrackTracker` keeps low-confidence detections for a second association
pass instead of throwing them away, improving continuity through short
occlusions without adding Ultralytics/AGPL code:

```python
from pyrobovision import ByteTrackTracker

tracker = ByteTrackTracker(high_threshold=0.5, low_threshold=0.1)
tracks = tracker.update(detections)
```

Only high-confidence detections create new tracks; low-confidence detections
may recover existing tracks. The implementation reuses the existing
Kalman/Hungarian primitives and is unit-tested with synthetic boxes.

## Bring your own detector (Supervision adapter)

PyRoboVision still does not ship a detector. It now accepts detections from
[Supervision](https://github.com/roboflow/supervision) (MIT), which normalizes
outputs from MMDetection, YOLO-NAS, and other permissive detectors:

```bash
pip install "pyrobovision[supervision]"
```

```python
from pyrobovision import SupervisionTrackerAdapter

adapter = SupervisionTrackerAdapter()
tracked = adapter.update(sv_detections)  # sv.Detections with tracker_id
```

`detections_from_supervision()` and `tracks_to_supervision()` are also exported
for custom pipelines. Ultralytics remains unsupported because it is AGPL-3.0.

## Alpamayo trajectory and reasoning adapter

[Alpamayo](https://github.com/NVlabs/alpamayo) (inference code Apache-2.0)
predicts ego trajectories and Chain-of-Causation reasoning. The optional
adapter normalizes its NumPy/tensor outputs without importing `torch` or
`alpamayo_r1`:

```python
from pyrobovision import AlpamayoTrajectoryAdapter

adapter = AlpamayoTrajectoryAdapter()  # 64 waypoints at 10 Hz
output = adapter.parse(pred_xyz, pred_rot, extra)  # upstream model outputs
best, metrics = adapter.select_best(output, ego_future_xyz)
```

`output.trajectories` exposes `(x, y, z)` waypoints, optional rotations, and
per-sample reasoning; `positions_2d()` drops `z` for 2D consumers. Alpamayo 1.5
and 2 preserve the same 64-waypoint/10 Hz contract. Synthetic CPU-only fixtures
for both versions live in `tests/fixtures/alpamayo/`. Alpamayo 1 weights are
research/non-commercial and inference requires an NVIDIA GPU, so model execution
stays outside this package.

## Installation

```bash
pip install pyrobovision

# real MiDaS depth estimation (installs PyTorch; one-time model download)
pip install "pyrobovision[depth]"

# ONNX model-export utilities
pip install "pyrobovision[onnx]"
```

Requires Python 3.10+. From source:

```bash
git clone https://github.com/Mullassery/PyRoboVision.git
cd PyRoboVision
pip install -e ".[dev]"
```

## Usage

Tracking plus trajectory prediction:

```python
import numpy as np
from pyrobovision.tracking.mot import MOTTracker, Detection
from pyrobovision.prediction.trajectory import TrajectoryPredictor

tracker = MOTTracker(max_age=30, min_hits=3)
positions = []
for x in (100, 110, 120, 130):
    tracker.update([Detection(bbox=np.array([x, 100, x + 40, 180]), confidence=0.9)])
    positions.append(tracker.get_track_by_id(1).get_position())

predictor = TrajectoryPredictor(model="cv")  # "cv" or "ca"
print(predictor.predict_trajectory(np.array(positions), horizon=5))
```

Depth + 3D boxes + occupancy (default heuristic depth; swap in `model="midas"` for real inference):

```python
import numpy as np
from pyrobovision.perception.depth import DepthEstimator
from pyrobovision.perception.bbox_3d import Box3DConverter
from pyrobovision.perception.occupancy import OccupancyGridBuilder

estimator = DepthEstimator(model="heuristic")
estimator.set_calibration(fx=500, fy=500, cx=320, cy=240)
depth = estimator.estimate_depth(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

box = Box3DConverter().from_2d_bbox_and_depth(
    bbox_2d=np.array([100, 100, 200, 200]), depth_map=depth.data, fx=500, fy=500, cx=320, cy=240)
grid = OccupancyGridBuilder(grid_size=(100, 100), resolution=0.1).from_3d_bboxes([box])
print(len(grid.get_occupied_cells()))
```

IMU/GPS fusion and safety checks:

```python
from pyrobovision.fusion.sensor_fusion import SensorFusionEngine, IMUData, GPSData
from pyrobovision.learning.safety import SafetyValidator

engine = SensorFusionEngine(origin_lat=37.7749, origin_lon=-122.4194)
engine.update_imu(IMUData(timestamp=0.0, accelerometer=[0, 0, 9.8], gyroscope=[0, 0, 0]))
state = engine.update_gps(GPSData(timestamp=0.1, latitude=37.7749, longitude=-122.4194,
                                  altitude=10.0, speed=5.0, heading=90.0, accuracy=2.0))

safe = SafetyValidator(max_acceleration=5.0, max_steering=45.0).correct_action(
    proposed_action=[6.5, 50.0], context={"speed": 15.0})  # clamped to limits
```

## Project status

Beta (Development Status 4). Current source version is **3.3.1**; the latest PyPI release is **3.3.1**. `tracking/` and `prediction/` are implemented and tested; supporting utilities (perception, fusion, learning) are real but minimal. **311 passed / 9 skipped** in the local suite; the skips are optional torch/MiDaS paths that run when the `[depth]` extra is installed. `black` / `mypy` are not enforced in CI today. See [docs/internal/ROADMAP_HONEST.md](docs/internal/ROADMAP_HONEST.md).

## For AI agents

Canonical facts for automated tooling and coding agents:

| Fact | Value |
|---|---|
| Package | `pyrobovision` |
| Python | `>=3.10` |
| Runtime deps | NumPy + SciPy only |
| Core entry points | `MOTTracker`, `ByteTrackTracker`, `TrajectoryPredictor`, `AlpamayoTrajectoryAdapter` |
| Source layout | `src/pyrobovision/{tracking,prediction,perception,fusion,learning,behavior,intent}` |
| Tests | `pip install -e ".[dev]"` then `pytest tests/ -v` |
| Coverage | `pytest tests/ --cov=src/pyrobovision --cov-report=term-missing` |
| Lint / format | `ruff check src/pyrobovision/`, `black src/ tests/`, `isort src/ tests/` |
| GPU needed? | **No** for tracker, predictor, adapters, or tests |
| Optional extras | `[depth]` (PyTorch + MiDaS), `[onnx]`, `[supervision]` |
| Alpamayo execution | **External**; adapter parses outputs only |
| SANA-WM | **Not integrated** |

Agent-friendly docs: [CLAUDE.md](./CLAUDE.md), [docs/internal/ROADMAP_HONEST.md](docs/internal/ROADMAP_HONEST.md), [docs/ALPAMAYO_WORKFLOW.md](./docs/ALPAMAYO_WORKFLOW.md).

## Documentation

- [examples/](examples/) — `basic_tracking.py`, `depth_estimation.py`, `real_detector_tracking.py`
- [CONTRIBUTING.md](./CONTRIBUTING.md) — development setup and guidelines
- [docs/internal/ROADMAP_HONEST.md](docs/internal/ROADMAP_HONEST.md) — done / not done / next, and technical debt
- [docs/OCCUPANCY.md](./docs/OCCUPANCY.md) — occupancy grids, raycasting, and line of sight
- [docs/ALPAMAYO_WORKFLOW.md](./docs/ALPAMAYO_WORKFLOW.md) — Alpamayo install and trajectory ingestion boundary
- [docs/BENCHMARKS.md](./docs/BENCHMARKS.md) — benchmark status and how to run your own
- [docs/CODE_OF_CONDUCT.md](./docs/CODE_OF_CONDUCT.md) — community guidelines
- [CHANGELOG.md](./CHANGELOG.md) — notable changes
- [SECURITY.md](./SECURITY.md) — vulnerability reporting

## Known issues and deferred work

**Blocked / failed:**
- Detector/GPU integration; otherwise freeze.

**Rejected / deferred:**
- Foundation models/CLIP/SAM.
- SANA-WM synthetic-video evaluation harness (not shipped; tracker remains frame-source agnostic).

Full shipped/blocked/rejected detail: [`docs/internal/QUEUE_STATUS.md`](docs/internal/QUEUE_STATUS.md).

## Contributing

Contributions welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md). Security issues: [SECURITY.md](./SECURITY.md).

## Part of the robotics & autonomy family

PyRoboVision is one of six complementary, composable tools in the Mullassery **robotics & autonomy** family. Together they form a single stack: **perceive → map → simulate → learn → debug → coordinate the fleet**.

| Tool | Role | What it does |
|---|---|---|
| [PyRoboVision](https://github.com/Mullassery/PyRoboVision) | Perception | **This project** — multi-object tracker (Kalman filter + Hungarian algorithm) and trajectory predictor for robot perception. |
| [PyTerrainMap](https://github.com/Mullassery/PyTerrainMap) | Mapping / SLAM | Fuses multi-robot terrain observations into a spatial-temporal store, with H3 spatial indexing and SLAM. |
| [PyRoboSimulator](https://github.com/Mullassery/PyRoboSimulator) | Simulation | World simulator for autonomous systems, with a FastAPI backend. |
| [PyRoboFrames](https://github.com/Mullassery/PyRoboFrames) | Learning data | Rust-backed ML dataloader for robot-learning datasets (native LeRobot format, Apple Silicon hardware video decode). |
| [PyRoboReplay](https://github.com/Mullassery/PyRoboReplay) | Mission forensics | Forensic-debugging and causal-analysis engine for autonomous robot mission replay. |
| [SyncFleet](https://github.com/Mullassery/SyncFleet) | Fleet coordination | Rust + SQLite, peer-to-peer state sync for robotics fleets — mutual-TLS 1.3, version-vector replication, mDNS discovery; conflict resolution on the roadmap. |

Each works standalone, but they're complementary by design. [Browse all projects →](https://github.com/Mullassery?tab=repositories)

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
