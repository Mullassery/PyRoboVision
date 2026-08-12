# Contributing to PyRoboVision

Thanks for your interest! PyRoboVision is a pure Python (NumPy + SciPy) multi-object
tracking and trajectory-prediction library, with supporting utilities for 3D
perception, behavior/intent classification, and simple imitation learning. See the
README for an honest breakdown of what's real vs. experimental.

## Project layout

```
src/pyrobovision/
├── tracking/     Kalman filter + Hungarian-algorithm multi-object tracking (the core)
├── prediction/   Trajectory forecasting (constant velocity/acceleration) + uncertainty
├── perception/   Depth estimation (real MiDaS via optional torch, or a placeholder
│                 heuristic), 3D bbox conversion, LiDAR utilities, occupancy grids
├── behavior/     Rule-based motion/behavior classification
├── intent/       Rule-based intent prediction
├── learning/     Imitation learning, behavior cloning, safety constraint validation
└── fusion/       IMU/GPS sensor fusion, ONNX export utilities (optional torch)

tests/            Tests for every module above (pytest)
examples/         Usage examples
```

## Dev setup

```bash
python3 -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Optional: real MiDaS depth backend / ONNX export utilities
pip install -e ".[dev,depth,onnx]"

# Run tests
pytest tests/ -v
```

**Requirements:**
- Python >= 3.10
- NumPy, SciPy (installed automatically — that's all the core package needs)
- PyTorch >= 2.6 only if you want `DepthEstimator(model="midas")` or the
  ONNX export helpers in `pyrobovision.fusion.optimization`

## Before opening a PR

- `black src/ tests/` and `isort src/ tests/`
- `mypy src/pyrobovision/` and `ruff check src/pyrobovision/`
- `pytest tests/ -v` passes (with coverage: `pytest tests/ -v --cov=pyrobovision`)
- New behavior has tests — no exceptions for the tracking/prediction core
- Claims in docstrings/README should be checked against what the code actually does;
  this project has previously shipped inaccurate performance/feature claims and we're
  trying hard not to repeat that

## High-impact areas

- **Real object detection integration** — the tracker takes detections as input but
  ships with no detector; examples/adapters for common detectors (YOLO, etc.) would help
- **Association robustness** — appearance features, better gating for fast-moving/occluded objects
- **Depth**: broader MiDaS variants, or a metric-scale calibration path
- **Trajectory prediction**: learned (not just constant velocity/acceleration) models

## License

By contributing, you agree your contributions are licensed under the [Proprietary License](./LICENSE).

## Questions?

Open a GitHub issue or discussion — we're here to help!
