# PyRoboVision Benchmarks

An earlier version of this file cited detailed FPS/latency/VRAM numbers for
`CylindricalStitcher`, `get_waymo_layout`, `MultiModalFusion`, and SAM3/CLIP/Grounding
DINO benchmarks. None of those classes or functions exist in this codebase — the
numbers were fabricated for code that was never built. They've been removed rather
than corrected, because there's nothing here to benchmark yet.

## What actually exists to benchmark

- `pyrobovision.tracking.mot.MOTTracker` (Kalman filter + Hungarian association)
- `pyrobovision.prediction.trajectory` (constant velocity/acceleration models)
- `pyrobovision.perception.depth.DepthEstimator` (real MiDaS_small via `torch.hub`,
  or the `heuristic` placeholder)
- `pyrobovision.perception.lidar` / `occupancy` (point cloud + occupancy grid utilities)

No formal benchmark suite has been run and published for these yet — no throughput or
latency numbers are quoted here because none have been independently measured. If you
need numbers for your use case, the tools to get them are standard:

```python
import time
import numpy as np
from pyrobovision.tracking.mot import MOTTracker, Detection

tracker = MOTTracker()
detections = [Detection(bbox=np.array([0, 0, 40, 40]), confidence=0.9)]

# Warm up
for _ in range(5):
    tracker.update(detections)

times = []
for _ in range(200):
    start = time.perf_counter()
    tracker.update(detections)
    times.append(time.perf_counter() - start)

print(f"Average: {np.mean(times) * 1000:.3f} ms")
print(f"P99: {np.percentile(times, 99) * 1000:.3f} ms")
```

Test suite correctness (not performance) is verified in CI: `pytest tests/ -v`.

If you run real benchmarks against your own workload and want to contribute numbers
back, please include the machine spec, Python/NumPy/SciPy versions, and the exact
script used — see [CONTRIBUTING.md](../CONTRIBUTING.md).
