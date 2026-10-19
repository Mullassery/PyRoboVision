# PyRoboVision

Multi-object tracking (Kalman filter + Hungarian association) and trajectory prediction for robotics — pure Python (NumPy + SciPy) core, with an optional real MiDaS monocular depth backend.

[![CI](https://github.com/Mullassery/PyRoboVision/actions/workflows/tests.yml/badge.svg)](https://github.com/Mullassery/PyRoboVision/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/pyrobovision)](https://pypi.org/project/pyrobovision/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

## What it is

PyRoboVision's real, tested core is a multi-object tracker: Kalman-filter state estimation plus Hungarian-algorithm association, with occlusion handling (tracks are predicted through detection gaps and re-associate on reacquisition) and constant-velocity / constant-acceleration trajectory prediction with uncertainty. Around it are supporting utilities: 3D perception (depth, 3D boxes, LiDAR, occupancy grids), rule-based behavior/intent classification, and minimal imitation-learning / behavior-cloning / safety-constraint building blocks. The core depends on NumPy and SciPy only; PyTorch is optional.

## Features

- **Multi-object tracking** — `MOTTracker`, `Track`, `Detection`; birth/confirm/occlude/death lifecycle; IoU association via `scipy.optimize.linear_sum_assignment`.
- **Occlusion handling** — every track is predicted each frame; a confirmed track survives up to `max_age` missed frames and re-associates to the same `track_id`.
- **Trajectory prediction** — `TrajectoryPredictor` with `"cv"` (constant velocity) or `"ca"` (constant acceleration) models and uncertainty.
- **Depth estimation** — `DepthEstimator(model="heuristic")` (default, no extra deps) or `model="midas"` (real pretrained MiDaS_small via `torch.hub`, optional `[depth]` extra).
- **3D perception utilities** — `Box3DConverter` (2D bbox + depth → 3D), LiDAR point-cloud filtering/clustering, BEV occupancy grids.
- **Sensor fusion** — Kalman-based IMU/GPS fusion with covariance (`SensorFusionEngine`).
- **ONNX export** — thin wrapper around `torch.onnx.export` (optional `[onnx]` extra).
- **Learning helpers** — imitation learning, behavior cloning, `SafetyValidator` constraint correction.

## Who it's for / Use cases

- **Adding tracking on top of your own detector** — feed bounding boxes from YOLO/SAM/anything into `MOTTracker`; it handles identity persistence through occlusion.
- **Forecasting from tracked positions** — constant-velocity/acceleration prediction with uncertainty for downstream collision-avoidance logic.
- **IMU/GPS fusion** — a fused position estimate with covariance without hand-rolling a Kalman filter.
- **Prototyping 3D perception** — depth maps, 3D boxes, and occupancy grids from 2D detections and a calibrated camera.

## What it is not

- **No object detector.** You bring your own detections (bounding boxes + confidence); none is bundled or wrapped.
- **No GPU-accelerated inference pipeline.** The tracking/prediction core is CPU NumPy/SciPy.
- **No foundation-model integration** (CLIP/SAM/Grounding DINO). An earlier version shipped a module that returned hardcoded fake results; it was removed, not fixed.
- **MiDaS output is relative, not metric** — correctly ordered near/far depth, not meters, unless you calibrate against a known reference.
- **Not a certified or safety-verified autonomous-driving stack.** `SafetyValidator` is a basic constraint checker.
- **The default `heuristic` depth backend is a Sobel-edge placeholder, not a learned depth model.**

## Project status

Beta (Development Status 4), version **3.1.0** (matching the latest PyPI release). `tracking/` and `prediction/` are implemented and tested; supporting utilities (perception, fusion, learning) are real but minimal. **278 passed / 7 skipped, 84% measured line coverage** on Python 3.11 (the 7 skips are the torch/MiDaS-dependent tests, which run when the `[depth]` extra is installed). `black --check` / `isort --check-only` currently report unformatted files and `mypy` reports pre-existing type errors, none enforced in CI today. See [ROADMAP_HONEST.md](./ROADMAP_HONEST.md).

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

## Quick start

```python
import numpy as np
from pyrobovision.tracking.mot import MOTTracker, Detection

tracker = MOTTracker(max_age=30, min_hits=3)

# Detections come from *your* detector — each is a bounding box + confidence.
detections = [Detection(bbox=np.array([100, 100, 140, 180]), confidence=0.92)]

for track in tracker.update(detections):
    print(track.track_id, track.get_position(), track.get_velocity())
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

## Documentation

- [examples/](examples/) — `basic_tracking.py`, `depth_estimation.py`, `real_detector_tracking.py`
- [CONTRIBUTING.md](./CONTRIBUTING.md) — development setup and guidelines
- [ROADMAP_HONEST.md](./ROADMAP_HONEST.md) — done / not done / next, and technical debt
- [docs/BENCHMARKS.md](./docs/BENCHMARKS.md) — benchmark status and how to run your own
- [docs/CODE_OF_CONDUCT.md](./docs/CODE_OF_CONDUCT.md) — community guidelines
- [CHANGELOG.md](./CHANGELOG.md) — notable changes
- [SECURITY.md](./SECURITY.md) — vulnerability reporting

## Queue status

Shipped, blocked/failed, and rejected work for this repository is documented in
[`QUEUE_STATUS.md`](QUEUE_STATUS.md).

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
