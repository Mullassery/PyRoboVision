# PyRoboVision Roadmap

**Current version:** 3.0.0

## What's done and real

- Multi-object tracking — Kalman filter + Hungarian algorithm association (`tracking/`)
- Occlusion handling — tracks survive and predict through missed-detection gaps, and
  re-associate on reacquisition (tested in `tests/test_occlusion.py`)
- Trajectory prediction — constant velocity/acceleration models with uncertainty
- Behavior/intent classification — rule-based, not learned
- 3D perception utilities — depth estimation (real MiDaS backend, optional), 3D
  bbox conversion, LiDAR point-cloud processing, occupancy grids
- Imitation learning / behavior cloning / safety-constraint validation building blocks
- IMU/GPS sensor fusion (Kalman-based)
- 277 tests passing, no dead/foreign-package tests mixed into the suite

## Known gaps (not built, not claimed as built)

- No bundled object detector — you supply detections (bounding boxes) to the tracker
- No GPU-accelerated inference pipeline
- No foundation-model integration (no CLIP/SAM/Grounding DINO wrapper — an earlier
  version of this project shipped one that only returned hardcoded fake results; it
  has been removed rather than left in place)
- MiDaS depth output is *relative* depth, not metric meters, unless separately
  calibrated
- Not a certified or safety-verified autonomous driving stack

## Near-term priorities

- [ ] Example adapter showing how to feed a real detector's output into `MOTTracker`
- [ ] Metric-scale calibration helper for the MiDaS depth backend
- [ ] Expand trajectory prediction beyond CV/CA (e.g. a small learned model, optional-torch)
- [ ] CI matrix covering the `depth` extra (currently skipped in CI to keep it fast/offline)

## Out of scope for now

Full detection -> tracking -> 3D -> planning -> safety "autonomous driving stack"
claims have been removed from this project's docs. If that's what you need, this
library can be one component (the tracking/prediction layer) of such a system, but
it is not one on its own.
