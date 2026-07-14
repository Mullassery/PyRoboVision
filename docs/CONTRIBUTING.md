# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

PyRoboVision is a pure Python library for autonomous driving perception and foundation model inference. It builds on [PyRoboFrames](https://github.com/Mullassery/PyRoboFrames) for data loading.

**Code structure** (`pyrobovision/`):
- `automotive/` — Autonomous driving algorithms
  - `stitching.py` — Cylindrical panoramic stitching (multi-camera 360° fusion)
  - `bev.py` — Bird's-eye-view projection (2D → 3D coordinate transform)
  - `fusion.py` — Sensor fusion (Lidar/Radar + occupancy grid)
  - `loaders.py` — Dataset readers (Waymo TFRecord, nuScenes JSON, KITTI stereo)

- `foundation/` — Multi-modal foundation models
  - `segmentation.py` — SAM3 instance segmentation + temporal tracking
  - `detection.py` — Grounding DINO open-vocabulary object detection
  - `embeddings.py` — CLIP scene understanding, text-image similarity
  - `fusion.py` — Multi-modal fusion (detection + segmentation + classification)

- `compute/` — Hardware acceleration abstraction
  - `gpu.py` — CuPy (NVIDIA GPU), MLX (Apple Silicon), NumPy (CPU) backends
  - Device auto-selection: tries CUDA → MLX → CPU

**Key design**: PyRoboVision is a **consumer library**, not a foundation. It consumes data from PyRoboFrames loaders and applies perception algorithms. No data loading logic here — that's PyRoboFrames' responsibility.

## Build & Test Commands

**Install**:
```bash
pip install "pyroboframes>=1.1.0" pyrobovision

# With GPU support
pip install "pyroboframes>=1.1.0" "pyrobovision[cuda]"
pip install "pyroboframes>=1.1.0" "pyrobovision[mlx]"

# From source (dev)
pip install -e ".[dev]"
```

**Tests**:
```bash
pytest                           # All tests
pytest tests/test_stitching.py  # Single file
pytest -v --cov=pyrobovision   # With coverage
```

**Format & lint**:
```bash
black .
isort .
mypy pyrobovision
ruff check .
```

**Run examples**:
```bash
python examples/panoramic_stitching.py
python examples/foundation_models.py
```

## Important Implementation Details

- **Stitching backend**: Uses cylindrical projection. Handles overlapping camera fields with Laplacian pyramid blending + graph-cut seams for seamless transitions.
- **BEV projection**: 3D LiDAR points → 2D orthographic projection. Configurable height slices (e.g., 0-2m for vehicle detection).
- **Device abstraction**: `compute.get_device()` auto-selects: CUDA if available, MLX on macOS, CPU fallback. Abstraction layer prevents device-specific code scattered throughout.
- **Foundation models**: SAM3/CLIP/Grounding DINO are inference-only wrappers around Hugging Face transformers. No training code.
- **PyRoboFrames dependency**: Minimum 1.1.0 required. Uses `prf.RoboFrameDataset` and hardware-decoded frames. Zero-copy when possible.
