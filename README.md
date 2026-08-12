# PyRoboVision

**Multi-object tracking (Kalman filter + Hungarian algorithm) and trajectory prediction, in pure Python.**

[![PyPI](https://img.shields.io/pypi/v/pyrobovision)](https://pypi.org/project/pyrobovision/)
[![Python](https://img.shields.io/pypi/pyversions/pyrobovision)](https://pypi.org/project/pyrobovision/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-277%20passing-brightgreen)](./tests)
[![Coverage](https://img.shields.io/badge/coverage-89%25-green)]()

## What this actually is

PyRoboVision's real, tested core is a **multi-object tracker**: Kalman filter state
estimation + Hungarian algorithm association, with occlusion handling (tracks are
predicted through gaps in detection and re-associate on reacquisition) and constant
velocity/acceleration trajectory prediction with uncertainty quantification.

Around that core there are supporting utilities: 3D perception (a real MiDaS
monocular depth backend, 3D bounding box conversion, LiDAR point-cloud processing,
occupancy grids), rule-based behavior/intent classification, and simple imitation
learning / behavior cloning / safety-constraint building blocks.

**This is not** a full detection -> tracking -> 3D -> planning -> safety autonomous
driving stack, and doesn't claim to be. See [What's NOT included](#whats-not-included)
below for specifics — an earlier version of this README oversold this project, and
we'd rather be boring and accurate than impressive and wrong.

The core package depends on **NumPy and SciPy only**. PyTorch is an optional extra,
needed only if you want real MiDaS depth inference or the ONNX export helpers.

---

## 30-Second Start

```python
import numpy as np
from pyrobovision.tracking.mot import MOTTracker, Detection

tracker = MOTTracker(max_age=30, min_hits=3)

# Detections come from *your* detector (YOLO, SAM, a color blob detector,
# whatever) — PyRoboVision does not ship one. Each is just a bounding box
# plus a confidence score.
detections = [Detection(bbox=np.array([100, 100, 140, 180]), confidence=0.92)]

confirmed_tracks = tracker.update(detections)
for track in confirmed_tracks:
    print(f"track {track.track_id}: position={track.get_position()}, velocity={track.get_velocity()}")
```

---

## Installation

```bash
pip install pyrobovision

# With real MiDaS depth estimation (installs PyTorch, one-time model download)
pip install "pyrobovision[depth]"

# With ONNX model export utilities
pip install "pyrobovision[onnx]"

# From source
git clone https://github.com/Mullassery/PyRoboVision.git
cd PyRoboVision
pip install -e ".[dev]"
```

---

## Quick Start

### Multi-object tracking + trajectory prediction

```python
import numpy as np
from pyrobovision.tracking.mot import MOTTracker, Detection
from pyrobovision.prediction.trajectory import TrajectoryPredictor

tracker = MOTTracker(max_age=30, min_hits=3)

# Simulate a few frames of detections for one object moving right.
boxes = [
    np.array([100.0, 100.0, 140.0, 180.0]),
    np.array([110.0, 100.0, 150.0, 180.0]),
    np.array([120.0, 100.0, 160.0, 180.0]),
    np.array([130.0, 100.0, 170.0, 180.0]),
]

positions = []
for box in boxes:
    tracker.update([Detection(bbox=box, confidence=0.9)])
    track = tracker.get_track_by_id(1)
    positions.append(track.get_position())

# Predict where the object goes next, using its tracked position history.
predictor = TrajectoryPredictor(model="cv")  # "cv" = constant velocity, "ca" = constant acceleration
future_positions = predictor.predict_trajectory(np.array(positions), horizon=5)
print(future_positions)
```

Occlusion is handled automatically: if a tracked object stops being detected for a
few frames (`tracker.update([])`), its Kalman filter keeps predicting its position
forward, and the track survives until `max_age` frames without a match — then a
detection reappearing near the predicted location re-associates to the *same*
`track_id`. This is exercised directly in
[`tests/test_occlusion.py`](./tests/test_occlusion.py).

### 3D perception: depth, 3D boxes, occupancy grid

```python
import numpy as np
from pyrobovision.perception.depth import DepthEstimator
from pyrobovision.perception.bbox_3d import Box3DConverter
from pyrobovision.perception.occupancy import OccupancyGridBuilder

# model="heuristic" (default) needs no extra dependencies but is a
# non-neural edge-density PLACEHOLDER, not real depth — see the module
# docstring in perception/depth.py. For genuine learned depth:
#   pip install "pyrobovision[depth]"
#   estimator = DepthEstimator(model="midas")
estimator = DepthEstimator(model="heuristic")
estimator.set_calibration(fx=500, fy=500, cx=320, cy=240)

rgb_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # your camera frame
depth_map = estimator.estimate_depth(rgb_frame)

converter = Box3DConverter()
bbox_3d = converter.from_2d_bbox_and_depth(
    bbox_2d=np.array([100, 100, 200, 200]),
    depth_map=depth_map.data,
    fx=500, fy=500, cx=320, cy=240,
)

builder = OccupancyGridBuilder(grid_size=(100, 100), resolution=0.1)
occupancy_grid = builder.from_3d_bboxes([bbox_3d])
print(f"Occupied cells: {len(occupancy_grid.get_occupied_cells())}")
```

### Sensor fusion (IMU/GPS) and safety-constrained learning

```python
from pyrobovision.fusion.sensor_fusion import SensorFusionEngine, IMUData, GPSData
from pyrobovision.learning.safety import SafetyValidator

fusion_engine = SensorFusionEngine(origin_lat=37.7749, origin_lon=-122.4194)
state = fusion_engine.update_imu(IMUData(timestamp=0.0, accelerometer=[0, 0, 9.8], gyroscope=[0, 0, 0]))
state = fusion_engine.update_gps(GPSData(latitude=37.7749, longitude=-122.4194, altitude=10.0,
                                          speed=5.0, heading=90.0, accuracy=2.0))
print(f"Fused position: {state.position}, uncertainty: {state.covariance}")

validator = SafetyValidator(max_acceleration=5.0, max_steering=45.0)
safe_action = validator.correct_action(proposed_action, {"speed": 15.0})
```

---

## What's real vs. placeholder

| Module | Status |
|---|---|
| `tracking/` (Kalman filter, Hungarian association, MOT) | **Real**, tested (`test_kalman_filter.py`, `test_association.py`, `test_mot_tracker.py`, `test_occlusion.py`) |
| `prediction/` (trajectory forecasting, uncertainty) | **Real**, tested |
| `perception/depth.py` with `model="midas"` | **Real** — genuine pretrained MiDaS_small inference via `torch.hub` (optional `[depth]` extra) |
| `perception/depth.py` with `model="heuristic"` (default) | **Placeholder** — Sobel-edge heuristic, not a depth model, documented as such in code |
| `perception/bbox_3d.py`, `lidar.py`, `occupancy.py` | Real geometric utilities (not neural), tested |
| `behavior/`, `intent/` | Rule-based classification, not learned models |
| `learning/` (imitation, behavior cloning, safety, training) | Real, minimal implementations; not benchmarked against production RL frameworks |
| `fusion/sensor_fusion.py` | Real Kalman-based IMU/GPS fusion |
| `fusion/optimization.py` (ONNX export) | Real, thin wrapper around `torch.onnx.export` (optional `[onnx]`/`[depth]` extra) |

## What's NOT included

- **No object detector.** You bring your own detections (bounding boxes + confidence)
  into `MOTTracker`. There is no YOLO/SAM/etc. bundled or wrapped here.
- **No GPU-accelerated inference pipeline.** The tracking/prediction core is plain
  NumPy/SciPy; it runs on CPU.
- **No foundation-model integration** (CLIP, SAM, Grounding DINO). An earlier version
  of this package shipped an "MCP 2.0" module and CLI/server "workflow" layer that
  claimed to do this — it only ever returned hardcoded fake results (e.g. a fixed
  `"frames_processed": 1200` regardless of input) and has been removed.
- **Not a certified or safety-verified autonomous driving stack.** `SafetyValidator`
  is a basic constraint-checking utility, not a certified safety system.
- **MiDaS depth output is relative, not metric.** `model="midas"` gives correctly
  *ordered* near/far depth, not depth in meters, unless you calibrate it against a
  known reference distance yourself.

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

277 tests passing, ~89% coverage, on Python 3.11 (no `torch` installed — the one test
that requires real MiDaS inference skips cleanly without it and runs when the
`[depth]` extra is installed). CI runs the same command on every push.

---

## Architecture

```
src/pyrobovision/
├── tracking/     Kalman filter + Hungarian-algorithm multi-object tracking
├── prediction/   Trajectory forecasting (CV/CA models) + uncertainty
├── perception/   Depth estimation, 3D bbox conversion, LiDAR, occupancy grids
├── behavior/     Rule-based motion/behavior classification
├── intent/       Rule-based intent prediction
├── learning/     Imitation learning, behavior cloning, safety constraints, training loop
└── fusion/       IMU/GPS sensor fusion, ONNX export utilities
```

---

## Related Projects

- **[PyRoboFrames](https://github.com/Mullassery/PyRoboFrames)** — a separate,
  independently published dataloader project by the same author (LeRobot, RLDS,
  HDF5, NetCDF, S3/GCS). PyRoboVision does not depend on it — if you want to feed
  PyRoboFrames-loaded data into this tracker, install it separately.

---

## Documentation

- [CONTRIBUTING.md](./CONTRIBUTING.md) — development setup and guidelines
- [ROADMAP.md](./ROADMAP.md) — what's done, what's not, what's next
- [SECURITY.md](./SECURITY.md) — vulnerability reporting
- [docs/CODE_OF_CONDUCT.md](./docs/CODE_OF_CONDUCT.md) — community guidelines
- [docs/BENCHMARKS.md](./docs/BENCHMARKS.md) — current benchmark status (none published yet — how to run your own)

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md).

For security issues, see [SECURITY.md](./SECURITY.md).

## License

Proprietary — Georgi Mammen Mullassery. See [LICENSE](./LICENSE).

## Citation

```bibtex
@software{mullassery2026pyrobovision,
  title={PyRoboVision: Multi-object tracking and trajectory prediction for robotics},
  author={Mullassery, Georgi},
  url={https://github.com/Mullassery/PyRoboVision},
  year={2026}
}
```
