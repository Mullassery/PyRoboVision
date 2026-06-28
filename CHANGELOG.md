# Changelog

All notable changes to PyRoboVision will be documented in this file.

## [0.5.0] — 2025-06-28

### Added (Phases 1–6)
- **Cylindrical panoramic stitching** — 360° multi-camera fusion for Waymo, nuScenes layouts
- **Laplacian pyramid blending** — Seamless multi-image blending with seam tracking
- **Bird's-eye-view (BEV) projection** — 3D perspective transformation for top-down perception
- **GPU acceleration** — CuPy (NVIDIA), MLX (Apple Silicon), NumPy (CPU) backends
- **Optical flow seam tracking** — Motion-aware panorama stitching
- **Multi-camera sensor fusion** — Waymo, nuScenes, KITTI dataset loaders
- **Lidar/Radar fusion** — Point cloud + camera integration with occupancy grids
- **Foundation models (Phase 7)**:
  - SAM3 temporal segmentation with tracking
  - CLIP scene embeddings and similarity search
  - Grounding DINO open-vocabulary object detection
  - Multi-modal fusion pipeline (detection + segmentation + classification)

### Fixed
- Dataset loader robustness for incomplete KITTI frames
- Edge case handling in panoramic stitching near seams

### Tests
- 149 tests passing across all phases
- Full coverage of stitching, blending, BEV, GPU acceleration, dataset loading, sensor fusion, and foundation models

## [0.4.0] — Phase 6 Foundation
- Lidar/Radar fusion infrastructure
- Occupancy grid mapping
- Sensor time-alignment utilities

## [0.3.0] — Phase 5 Foundation
- Waymo/nuScenes/KITTI dataset loaders
- Multi-camera synchronization

## [0.2.0] — Phase 4 Foundation
- GPU acceleration backends (CuPy, MLX)
- Optical flow seam tracking

## [0.1.0] — Phases 1–3 Foundation
- Cylindrical stitching and BEV
- Laplacian blending
- NumPy backend

---

## Roadmap

### Near-term (Phase 8)
- Real-time panoramic stitching optimization (streaming friendly)
- ONNX/TensorRT export for foundation models
- End-to-end benchmark suite (throughput, latency, memory)

### Medium-term (Phase 9)
- Multi-view 3D reconstruction
- Temporal tracking across frames
- Ego-motion estimation

### Long-term (Phase 10+)
- Closed-loop control pipelines
- Live deployment guides (ROS, Docker)
- Mobile device optimization
