# Migration Guide: PyRoboFrames v0.5 → PyRoboVision v0.5

## Overview

PyRoboFrames v1.0 refocuses on its core mission: **fast dataloader for robot learning**. Autonomous driving perception and foundation models have been moved to a separate library: **PyRoboVision**.

This separation allows:
- PyRoboFrames to stay lean and focused
- PyRoboVision to evolve independently
- Clearer maintenance boundaries

## What Moved?

### ❌ Removed from PyRoboFrames v1.0

**Autonomous Driving (v0.5.x):**
```python
# ❌ These imports no longer work in PyRoboFrames
from pyroboframes.automotive import CylindricalStitcher
from pyroboframes.automotive import BEVProjector
from pyroboframes.automotive import LidarFusion
from pyroboframes.automotive import WaymoDatasetLoader
from pyroboframes.automotive import NuScenesDatasetLoader
# ... etc
```

**Foundation Models (Phase 7):**
```python
# ❌ These imports no longer work in PyRoboFrames
from pyroboframes.automotive import SAM3Segmenter
from pyroboframes.automotive import CLIPEmbedding
from pyroboframes.automotive import GroundingDINO
from pyroboframes.automotive import MultiModalFusion
```

### ✅ Still in PyRoboFrames v1.0

**Core Robot Learning:**
```python
# ✅ All of these still work in PyRoboFrames v1.0+
from pyroboframes import RoboFrameDataset
from pyroboframes import ProprioceptiveLoader      # P0 (new)
from pyroboframes import DataLoader
from pyroboframes import sensor_fusion
from pyroboframes import depth_io
from pyroboframes.dataframe import RoboticsDataFrame
```

---

## Migration Steps

### Step 1: Install PyRoboVision

```bash
# Install PyRoboVision (pulls PyRoboFrames as dependency)
pip install PyRoboVision

# Or from source
git clone https://github.com/Mullassery/PyRoboVision.git
cd PyRoboVision
pip install -e .
```

### Step 2: Update Imports

**Before (PyRoboFrames v0.5):**
```python
from pyroboframes.automotive import (
    CylindricalStitcher,
    BEVProjector,
    LidarFusion,
    SAM3Segmenter,
    CLIPEmbedding,
)
```

**After (PyRoboFrames v1.0 + PyRoboVision v0.5):**
```python
# Data loading (PyRoboFrames)
from pyroboframes import RoboFrameDataset, DataLoader

# Perception (PyRoboVision)
from robotics_perception.automotive import (
    CylindricalStitcher,
    BEVProjector,
    LidarFusion,
)
from robotics_perception.foundation_models import (
    SAM3Segmenter,
    CLIPEmbedding,
    MultiModalFusion,
)
```

### Step 3: Update Code Examples

**Before:**
```python
import pyroboframes as prf

# Load dataset
ds = prf.RoboFrameDataset.from_path("...")

# Stitch
from pyroboframes.automotive import CylindricalStitcher, get_waymo_layout
layout = get_waymo_layout()
stitcher = CylindricalStitcher(layout)
```

**After:**
```python
import pyroboframes as prf
from robotics_perception.automotive import CylindricalStitcher, get_waymo_layout

# Load dataset (same as before)
ds = prf.RoboFrameDataset.from_path("...")

# Stitch (now from PyRoboVision)
layout = get_waymo_layout()
stitcher = CylindricalStitcher(layout)
```

---

## API Compatibility

### Good News: No API Changes

The code is identical—only the import path changed. All function signatures, behavior, and tests are the same.

```python
# ✅ These work exactly as before, just different import
from robotics_perception.automotive import CylindricalStitcher

layout = get_waymo_layout()
stitcher = CylindricalStitcher(layout, blend_method="laplacian")
panorama = stitcher.stitch(frames)  # Same as before
```

### Breaking Changes: None

There are no breaking changes. The APIs are identical. Only imports moved.

---

## Common Migration Patterns

### Pattern 1: Autonomous Driving Perception

**Old (PyRoboFrames v0.5):**
```python
from pyroboframes.automotive import (
    CylindricalStitcher,
    BEVProjector,
    get_waymo_layout,
)

layout = get_waymo_layout()
stitcher = CylindricalStitcher(layout)
```

**New (PyRoboFrames v1.0 + PyRoboVision):**
```python
from robotics_perception.automotive import (
    CylindricalStitcher,
    BEVProjector,
    get_waymo_layout,
)

layout = get_waymo_layout()
stitcher = CylindricalStitcher(layout)
```

### Pattern 2: Foundation Models

**Old (PyRoboFrames v0.5):**
```python
from pyroboframes.automotive import (
    SAM3Segmenter,
    CLIPEmbedding,
    GroundingDINO,
)

segmenter = SAM3Segmenter(device="mlx")
clip = CLIPEmbedding(device="mlx")
detector = GroundingDINO(device="mlx")
```

**New (PyRoboFrames v1.0 + PyRoboVision):**
```python
from robotics_perception.foundation_models import (
    SAM3Segmenter,
    CLIPEmbedding,
    GroundingDINO,
)

segmenter = SAM3Segmenter(device="mlx")
clip = CLIPEmbedding(device="mlx")
detector = GroundingDINO(device="mlx")
```

### Pattern 3: Mixed Robot Learning + Perception

**New capability: Combine both libraries**

```python
import pyroboframes as prf
from robotics_perception.automotive import MultiModalFusion

# Load robot learning data (PyRoboFrames)
ds = prf.RoboFrameDataset.from_path("waymo_dataset")
loader = prf.DataLoader(ds, cameras=["observation.images.front"])

# Apply perception (PyRoboVision)
fusion = MultiModalFusion(device="mlx")

for batch in loader:
    frames = batch["observation.images.front"]
    scene = fusion.understand(frames)
    print(f"Scene: {scene.scene_type}")
```

---

## Dependency Changes

### PyRoboFrames v1.0 Dependencies

```toml
# Smaller, focused dependencies
numpy>=1.24.0
torch>=2.0.0          # For dataloader device selection
scipy>=1.10.0
transformers>=4.30.0  # For tokenizers only
```

### PyRoboVision v0.5 Dependencies

```toml
# Depends on PyRoboFrames v1.0+
pyroboframes>=1.0.0

# Perception dependencies
torch>=2.0.0
transformers>=4.30.0  # For SAM3, CLIP, Grounding DINO
scipy>=1.10.0
open3d>=0.17.0        # For lidar point clouds

# Optional
cupy>=12.0.0          # For NVIDIA GPU acceleration
mlx>=0.0.13           # For Apple Silicon
```

---

## Testing

### PyRoboFrames v1.0 Tests

```bash
cd pyroboframes
pytest tests/test_robot_learning_* -v
pytest tests/test_proprioceptive_loader.py -v
```

### PyRoboVision v0.5 Tests

```bash
cd PyRoboVision
pytest tests/test_automotive_* -v
pytest tests/test_foundation_* -v
```

---

## FAQ

**Q: Do I have to migrate?**  
A: If you're using v0.5 (AV) or Phase 7 (foundation models), yes. PyRoboFrames v1.0 removed them. But installation is easy: `pip install PyRoboVision`.

**Q: Will my code break?**  
A: Only the import paths change. Everything else is identical. Simple find-and-replace should fix it.

**Q: Can I use both libraries together?**  
A: Yes! That's the whole point. Use PyRoboFrames v1.0 for data loading, PyRoboVision for perception algorithms.

**Q: Is PyRoboVision maintained?**  
A: Yes, by the same maintainer. Independent release cycle, same quality standards.

**Q: What about PyRoboFrames v0.5.x?**  
A: v0.5.x will be archived but remains available on GitHub and PyPI for backward compatibility.

**Q: When was this split?**  
A: PyRoboFrames v1.0 (released [date]). PyRoboVision v0.5.0 (released same time).

---

## Getting Help

- **PyRoboFrames issues:** https://github.com/Mullassery/PyRoboFrames/issues
- **PyRoboVision issues:** https://github.com/Mullassery/PyRoboVision/issues
- **Discussion:** Use the appropriate repo's discussions tab

---

## Summary

| Library | Purpose | Version | Status |
|---------|---------|---------|--------|
| **PyRoboFrames** | Fast ML dataloader for robot learning | v1.0+ | Active, stable |
| **PyRoboVision** | AV perception + foundation models | v0.5+ | Active, experimental |

**Both libraries complement each other. Use PyRoboFrames for data, PyRoboVision for perception.**
