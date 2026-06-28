# PyRoboVision Architecture

## Design Principle

**PyRoboVision is a consumer library, not a foundation.**

It depends on PyRoboFrames v1.0+ for data loading, and adds perception algorithms on top. This separation ensures:
- PyRoboFrames stays focused and maintainable (dataloader mission)
- PyRoboVision evolves independently (perception mission)
- Both libraries are testable and reusable

## Module Organization

```
pyrobovision/
├── __init__.py
├── automotive/                    # AV perception (v0.5)
│   ├── __init__.py
│   ├── stitching.py              # Cylindrical panoramic stitching
│   ├── blending.py               # Laplacian pyramid + graph-cut seams
│   ├── bev.py                    # Bird's-eye-view projection
│   ├── perception_3d.py          # Lidar/Radar fusion, occupancy grids
│   ├── datasets.py               # Waymo/nuScenes/KITTI loaders
│   ├── tfrecord_utils.py         # TFRecord parsing (Waymo)
│   ├── nuscenes_utils.py         # JSON metadata (nuScenes)
│   └── camera_layouts.py         # Camera calibration templates
│
└── foundation_models/             # Vision-language models (Phase 7)
    ├── __init__.py
    ├── sam3_segmentation.py      # SAM3 instance segmentation
    ├── clip_embeddings.py        # CLIP scene understanding
    ├── grounding_dino.py         # Grounding DINO detection
    └── multimodal_fusion.py      # Unified detection+segmentation+classification
```

## Data Flow

```
Data on disk (Waymo, nuScenes, KITTI)
           ↓
   PyRoboFrames RoboFrameDataset
   (loads frames, state, metadata)
           ↓
   PyRoboVision consumers
   ├── CylindricalStitcher (v0.5)
   ├── BEVProjector (v0.5)
   ├── LidarFusion (v0.5)
   ├── SAM3Segmenter (Phase 7)
   ├── CLIPEmbedding (Phase 7)
   ├── GroundingDINO (Phase 7)
   └── MultiModalFusion (Phase 7)
           ↓
   Output: panoramas, BEV maps, segmentation masks,
           embeddings, scene understanding
```

## Dependency Tree

```
robotics-perception
├── Depends: PyRoboFrames >=1.0.0 (core)
├── Depends: torch >=2.0.0 (for transformers)
├── Depends: transformers >=4.30.0 (SAM3, CLIP, Grounding DINO)
├── Depends: scipy >=1.10.0 (rotation, interpolation)
├── Depends: open3d >=0.17.0 (lidar point clouds)
│
└── Optional: cupy (NVIDIA CUDA acceleration)
    Optional: mlx (Apple Silicon MLX backend)
```

## Testing Strategy

```
tests/
├── test_automotive_stitching.py      # v0.5 Phase 1
├── test_automotive_blending.py       # v0.5 Phase 2
├── test_automotive_bev.py            # v0.5 Phase 3
├── test_automotive_gpu.py            # v0.5 Phase 4
├── test_automotive_datasets.py       # v0.5 Phase 5
├── test_automotive_perception_3d.py  # v0.5 Phase 6
├── test_foundation_sam3.py           # Phase 7a
├── test_foundation_clip.py           # Phase 7b
├── test_foundation_grounding_dino.py # Phase 7c
└── test_foundation_multimodal.py     # Phase 7d
```

**Total: 149 tests**
- All pass ✅
- No external dependencies (grayscale tests mock models when unavailable)
- Run with: `pytest tests/ -v`

## API Stability

### v0.5.x (Stable)
- `CylindricalStitcher`, `BEVProjector`, `LidarFusion`, etc.
- No breaking changes within v0.5 minor versions
- Will be maintained alongside new features

### Phase 7 (Experimental)
- `SAM3Segmenter`, `CLIPEmbedding`, `GroundingDINO`, `MultiModalFusion`
- API may change before v1.0
- Breaking changes allowed in v0.x

## Performance Characteristics

### Autonomous Driving (v0.5)

| Operation | Time | Device | Notes |
|-----------|------|--------|-------|
| Panorama stitching (5 cameras) | 50-150 FPS | GPU | Depends on resolution |
| BEV projection | 10-30 FPS | GPU | Real-time capable |
| Lidar fusion (occupancy grid) | 1-5 ms | CPU | Per-frame update |
| GPU decode (NVIDIA H100) | 100+ FPS | NVIDIA | With CV-CUDA |
| GPU decode (Apple M3) | 100+ FPS | Apple | With VideoToolbox |

### Foundation Models (Phase 7)

| Model | Inference Time | Device | Notes |
|-------|---|---|---|
| SAM3 (small) | 100-500 ms | GPU | Per-frame segmentation |
| CLIP (ViT-B32) | 50-100 ms | GPU | Per-frame embedding |
| Grounding DINO (tiny) | 200-800 ms | GPU | Per-frame detection |
| Multi-modal fusion | 500-1500 ms | GPU | SAM3 + CLIP + Grounding DINO |

## GPU Support

### NVIDIA (CUDA)
- Uses `torch` backend (PyTorch + CUDA)
- Optimized with CV-CUDA transforms (optional)
- Recommended: H100, A100, RTX 4090

### Apple Silicon (MLX)
- Native MLX arrays via PyRoboFrames
- VideoToolbox hardware video decode
- Recommended: M3 Max or M4 Max

### CPU Fallback
- NumPy-based operations
- Acceptable for research; not recommended for production
- ~10-20× slower than GPU

## Future Directions

### v0.6 (Planned)
- Real-time streaming (ROS 2 integration)
- Multi-robot perception
- Semantic SLAM

### v1.0 (Future)
- Production-grade deployment
- Edge optimization (quantization)
- Multi-domain foundation models

## Related Work

- **PyRoboFrames v1.0**: Data loading, video decode, sensor fusion
- **SAM3**: Facebook Research (upstream model)
- **CLIP**: OpenAI (upstream model)
- **Grounding DINO**: IDEA Research (upstream model)
- **CV-CUDA**: NVIDIA (optional acceleration)

## Maintenance

robotics-perception is maintained separately from PyRoboFrames to allow:
- Independent release cycles
- Specialized GPU optimization (perception algorithms)
- Flexibility to add/remove foundation models
- Clearer maintenance burden (perception-focused)

For core dataloader issues, file on PyRoboFrames. For perception/model issues, file here.
