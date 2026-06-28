# PyRoboVision Benchmarks

Performance metrics for autonomous driving perception and vision-language foundation models.

## Core Metrics

### Panoramic Stitching Performance
Measures throughput for cylindrical projection and blending across hardware backends.

| Backend | Resolution | FPS | Latency (ms) | Memory |
|---------|-----------|-----|--------------|--------|
| **MLX** (Apple Silicon M3) | 1024×768 | 45–60 | 16–22 | ~2GB |
| **CuPy** (RTX 4090) | 1024×768 | 120–150 | 6–8 | ~4GB VRAM |
| **NumPy** (CPU baseline) | 1024×768 | 5–8 | 120–200 | ~1GB |

### BEV Projection Performance
Top-down perspective transformation for autonomous driving perception.

| Operation | Time (ms) | Backend |
|-----------|-----------|---------|
| Single BEV projection | 5–8 | MLX |
| Batch BEV (8 images) | 15–20 | MLX |
| Occupancy grid fusion | 20–30 | MLX |

### Foundation Model Inference
Vision-language models on V100/H100 with batch=1, FP32.

| Model | Resolution | Latency (ms) | Memory | Throughput (img/s) |
|-------|-----------|--------------|--------|-------------------|
| **SAM3** (segmentation) | 1024×768 | 200–250 | 8GB VRAM | 4–5 |
| **CLIP** (embeddings) | 224×224 | 20–30 | 4GB VRAM | 33–50 |
| **Grounding DINO** (detection) | 640×480 | 150–200 | 6GB VRAM | 5–7 |
| **Multi-modal fusion** | 1024×768 | 450–600 | 12GB VRAM | 1.5–2 |

### Dataset Loading
Throughput for Waymo, nuScenes, and KITTI datasets.

| Dataset | Resolution | Frames/sec | Frames loaded/sec |
|---------|-----------|-----------|------------------|
| Waymo TFRecord | 1920×1280 | 25–30 | 200–250 |
| nuScenes JSON | 1600×900 | 35–45 | 280–360 |
| KITTI stereo | 1242×375 | 50–60 | 400–480 |

## Running Your Own Benchmarks

### Installation

```bash
pip install pyrobovision[dev]
```

### Stitching Performance

```python
import numpy as np
import time
from pyrobovision.automotive import CylindricalStitcher, get_waymo_layout

# Create test frames
layout = get_waymo_layout()
stitcher = CylindricalStitcher(layout, device="mlx")

frames = {k: np.random.randint(0, 256, (1280, 1024, 3), dtype=np.uint8)
          for k in layout.camera_names}

# Warm up
for _ in range(5):
    _ = stitcher.stitch(frames)

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    _ = stitcher.stitch(frames)
    times.append(time.time() - start)

print(f"Average: {np.mean(times)*1000:.1f}ms")
print(f"P99: {np.percentile(times, 99)*1000:.1f}ms")
print(f"FPS: {1/np.mean(times):.1f}")
```

### Foundation Model Inference

```python
from pyrobovision.foundation_models import MultiModalFusion
import numpy as np
import time

fusion = MultiModalFusion(device="cuda")
frame = np.random.randint(0, 256, (1024, 768, 3), dtype=np.uint8)

# Warm up
for _ in range(3):
    _ = fusion.understand(frame)

# Benchmark
times = []
for _ in range(50):
    start = time.time()
    scene = fusion.understand(frame)
    times.append(time.time() - start)

print(f"Average inference: {np.mean(times)*1000:.1f}ms")
print(f"Throughput: {1/np.mean(times):.1f} img/s")
```

## System Requirements for Benchmarking

- **Apple Silicon:** MLX 0.13+, 8GB+ RAM
- **NVIDIA:** CuPy 12+, 4GB+ VRAM (8GB+ for foundation models)
- **CPU:** NumPy only, 16GB+ RAM

## Expected Scaling

- **Resolution scaling:** Latency scales ~O(H×W) for stitching
- **Batch scaling:** ~Linear throughput gain with batch size up to GPU memory limit
- **Multi-camera:** Linear scaling with number of cameras (4–8 on typical autonomous vehicles)

## Profiling

To identify bottlenecks:

```python
import cProfile
from pyrobovision.automotive import CylindricalStitcher

profiler = cProfile.Profile()
profiler.enable()

# ... your benchmark code ...

profiler.disable()
profiler.print_stats(sort='cumulative')
```
