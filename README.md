# PyRoboVision

[![PyPI](https://img.shields.io/pypi/v/pyrobovision)](https://pypi.org/project/pyrobovision/)
[![Python](https://img.shields.io/pypi/pyversions/pyrobovision)](https://pypi.org/project/pyrobovision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-149%20passing-brightgreen)]()
[![PyRoboFrames](https://img.shields.io/badge/PyRoboFrames-1.1.0+-blue)](https://github.com/Mullassery/PyRoboFrames)

Advanced autonomous driving perception and vision-language foundation models for robotics. Builds on [PyRoboFrames 1.1.0+](https://github.com/Mullassery/PyRoboFrames) for data loading.

**Focus:** Perception algorithms (stitching, 3D fusion, foundation models) that consume data from PyRoboFrames or your own loaders.

**Note:** PyRoboVision is a **consumer library**, not a foundation. It handles perception — PyRoboFrames handles data loading. Clear separation of concerns.

---

## What's Inside

### Autonomous Driving (v0.5)
- **Cylindrical panoramic stitching** — 360° multi-camera fusion (Waymo, nuScenes)
- **Advanced blending** — Laplacian pyramid + graph-cut seams
- **Bird's-eye-view (BEV)** — 3D projection for autonomous perception
- **GPU acceleration** — CuPy (NVIDIA), MLX (Apple Silicon), NumPy (CPU)
- **Sensor fusion** — Lidar/Radar + occupancy grid mapping
- **Dataset loaders** — Waymo TFRecord, nuScenes JSON, KITTI stereo

### Foundation Models (Phase 7)
- **SAM3 segmentation** — Instance segmentation + temporal tracking
- **CLIP embeddings** — Scene understanding, text-image similarity
- **Grounding DINO** — Open-vocabulary object detection
- **Multi-modal fusion** — Unified detection + segmentation + classification

---

## Installation

```bash
# Requires PyRoboFrames 1.1.0+
pip install "pyroboframes>=1.1.0" pyrobovision

# With NVIDIA GPU support
pip install "pyroboframes>=1.1.0" "pyrobovision[cuda]"

# With Apple Silicon (MLX)
pip install "pyroboframes>=1.1.0" "pyrobovision[mlx]"

# From source
git clone https://github.com/Mullassery/PyRoboVision.git
cd PyRoboVision
pip install -e .
```

---

## Quick Start

### Autonomous Driving: 360° Panoramic Perception

```python
from pyrobovision.automotive import (
    CylindricalStitcher,
    get_waymo_layout,
)

# Stitch 5 cameras into 360° panorama
layout = get_waymo_layout()
stitcher = CylindricalStitcher(layout, blend_method="laplacian")

frames = {
    "FRONT": ...,
    "FRONT_LEFT": ...,
    # ... other cameras
}

panorama = stitcher.stitch(frames)  # [1, H, W, 3] seamless 360°
```

### Foundation Models: Multi-Modal Scene Understanding

```python
from pyrobovision.foundation_models import MultiModalFusion

fusion = MultiModalFusion(
    detection_prompt="car . pedestrian . cyclist",
    device="mlx",  # or "cuda"
)

scene = fusion.understand(frame)
for obj in scene.objects:
    print(f"{obj.object_class}: {obj.semantic_label}")
```

### Loading Data with PyRoboFrames 1.1.0

PyRoboVision consumes data loaded by PyRoboFrames. With 1.1.0 you can now load from
HDF5, NetCDF, RLDS, or stream from S3/GCS before passing frames to PyRoboVision:

```python
import pyroboframes as prf
from pyrobovision.automotive import CylindricalStitcher, get_waymo_layout

# Load a Waymo RLDS dataset and run panoramic stitching
prf.convert_rlds("waymo_open_dataset", "/tmp/waymo_lerobot")
ds = prf.RoboFrameDataset.from_path("/tmp/waymo_lerobot")

stitcher = CylindricalStitcher(get_waymo_layout(), blend_method="laplacian")

loader = ds.loader(
    cameras=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
    batch_size=1,
    output="numpy",
)

for batch in loader:
    frames = {cam: batch[cam][0] for cam in loader.cameras}
    panorama = stitcher.stitch(frames)
```

---

## Architecture

### Dependency Graph

```
PyRoboVision/
├── automotive/          # v0.5 AV perception
│   ├── stitching.py
│   ├── blending.py
│   ├── bev.py
│   ├── perception_3d.py
│   ├── tfrecord_utils.py
│   ├── nuscenes_utils.py
│   └── datasets.py
│
└── foundation_models/   # Phase 7
    ├── sam3_segmentation.py
    ├── clip_embeddings.py
    ├── grounding_dino.py
    └── multimodal_fusion.py

↓ Depends on PyRoboFrames 1.1.0+ (dataloader)
PyRoboFrames 1.1.0/
├── RoboFrameDataset      # Load LeRobot, HDF5, NetCDF, RLDS
├── ProprioceptiveLoader  # Load state/action only
├── DataLoader            # Device selection + caching
├── RemoteDataset         # S3/GCS streaming
├── DatasetValidator      # Data quality checks
└── [codec selection, quality scoring, distributed, ...]
```

**Key design:** PyRoboVision handles perception; PyRoboFrames handles data loading.
Any data source PyRoboFrames can load — LeRobot, RLDS, HDF5, NetCDF, S3/GCS — is
immediately usable as input to PyRoboVision algorithms.

---

## Features

| Phase | Feature | Status | Tests |
|---|---|---|---|
| **1** | Cylindrical panoramic projection | ✅ | 10 |
| **2** | Laplacian pyramid blending | ✅ | 5 |
| **3** | Bird's-eye-view (BEV) projection | ✅ | 5 |
| **4a** | GPU acceleration (CuPy/MLX/NumPy) | ✅ | 6 |
| **4b** | Optical flow seam tracking | ✅ | 10 |
| **5** | Waymo/nuScenes/KITTI loaders | ✅ | 9 |
| **6** | Lidar/Radar fusion + Occupancy grids | ✅ | 18 |
| **7a** | SAM3 temporal segmentation | ✅ | 18 |
| **7b** | CLIP scene embeddings | ✅ | 25 |
| **7c** | Grounding DINO detection | ✅ | 26 |
| **7d** | Multi-modal fusion | ✅ | 17 |

**Total: 149 tests, all passing**

---

## Use Cases

### Autonomous Driving
- Waymo/nuScenes perception pipeline (panoramic stitching + 3D fusion)
- Real-time BEV mapping from multi-camera rigs
- Open X-Embodiment dataset analysis via PyRoboFrames RLDS loader

### Mobile Manipulation
- Egocentric robot perception (360° view from mobile base)
- Scene understanding for pick-and-place

### Robotdog Navigation
- Panoramic localization from multi-camera fusion
- Terrain classification from BEV projection

### Data Pipeline Integration
- Validate incoming camera data before perception (`prf.DatasetValidator`)
- Load from remote S3/GCS warehouses into perception pipelines (`prf.RemoteDataset`)
- Convert legacy HDF5 simulation data for scene understanding (`prf.convert_hdf5`)

---

## Related Projects

- **[PyRoboFrames 1.1.0](https://github.com/Mullassery/PyRoboFrames)** — Fast ML dataloader (core dependency): LeRobot, RLDS, HDF5, NetCDF, S3/GCS, Ray
- **[LeRobot](https://github.com/huggingface/lerobot)** — HuggingFace robotics datasets
- **[Open X-Embodiment](https://robotics-transformer-x.github.io/)** — Cross-embodiment robotics datasets
- **[Segment Anything 3 (SAM3)](https://github.com/facebookresearch/segment-anything-3)** — Instance segmentation
- **[CLIP](https://github.com/openai/CLIP)** — Vision-language models
- **[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)** — Open-vocabulary detection

---

## Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Design and implementation
- [CONTRIBUTING.md](./CONTRIBUTING.md) — Development setup and guidelines
- [CHANGELOG.md](./CHANGELOG.md) — Version history
- [SECURITY.md](./SECURITY.md) — Vulnerability reporting
- [docs/BENCHMARKS.md](./docs/BENCHMARKS.md) — Performance benchmarks

---

## Community

- **GitHub Issues** — [Ask questions, report bugs](https://github.com/Mullassery/PyRoboVision/issues)
- **GitHub Discussions** — [Share ideas and best practices](https://github.com/Mullassery/PyRoboVision/discussions)
- **Code of Conduct** — [Be respectful and constructive](./CODE_OF_CONDUCT.md)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and guidelines.

For security issues, see [SECURITY.md](./SECURITY.md).

---

## License

MIT (same as PyRoboFrames) — © Georgi Mammen Mullassery

---

## Citation

```bibtex
@software{mullassery2025pyrobovision,
  title={PyRoboVision: Advanced perception and vision-language models for robotics},
  author={Mullassery, Georgi},
  url={https://github.com/Mullassery/PyRoboVision},
  year={2025}
}
```
