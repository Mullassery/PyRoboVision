# PyRoboVision Roadmap

**Current version:** 3.1.0

## What's done and real

- Multi-object tracking — Kalman filter + Hungarian algorithm association (`tracking/`)
- Occlusion handling — tracks survive and predict through missed-detection gaps, and
  re-associate on reacquisition (tested in `tests/test_occlusion.py`)
- Trajectory prediction — constant velocity/acceleration models with uncertainty, plus an
  optional learned `LearnedTrajectoryModel` (GRU, optional-torch) for curved/non-CV motion
- Behavior/intent classification — rule-based, not learned
- 3D perception utilities — depth estimation (real MiDaS backend, optional), 3D
  bbox conversion, LiDAR point-cloud processing, occupancy grids
- Imitation learning / behavior cloning / safety-constraint validation building blocks
- IMU/GPS sensor fusion (Kalman-based)
- 285 tests total (277 base + 8 added for `LearnedTrajectoryModel`), no dead/foreign-package
  tests mixed into the suite. Re-verified directly for this pass: `pytest tests/ -q` gives
  278 passed / 7 skipped without the optional `torch`/`depth` extra installed; the 7 skips
  are the MiDaS/torch-dependent tests, which CI's separate `test-depth-extra` job installs
  and runs for real (see `.github/workflows/tests.yml`) rather than leaving permanently skipped.

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

- [x] Example adapter showing how to feed a real detector's output into `MOTTracker` —
      `examples/real_detector_tracking.py`, torchvision Faster R-CNN (COCO-pretrained,
      real weights) feeding real per-frame detections into `MOTTracker`; verified end-to-end
      locally (real model download + inference + tracking, including a genuine missed-detection
      frame the tracker predicts through).
- [ ] Metric-scale calibration helper for the MiDaS depth backend
- [x] Expand trajectory prediction beyond CV/CA (e.g. a small learned model, optional-torch) —
      `LearnedTrajectoryModel` in `prediction/trajectory.py` (small GRU on velocity sequences,
      real Adam+MSE training via `fit()`, raises rather than returning garbage if used unfitted).
      Verified it actually generalizes: trained on circular trajectories at radii 8/10/12/15,
      it tracks a held-out radius-11 trajectory with ~11x lower error than `ConstantVelocityModel`
      (`tests/test_trajectory_prediction.py::TestLearnedTrajectoryModel`).
- [x] CI matrix covering the `depth` extra (currently skipped in CI to keep it fast/offline) —
      added a `test-depth-extra` job in `.github/workflows/tests.yml` installing
      `pyrobovision[depth]`. Along the way found `depth` was missing `timm` (MiDaS_small's
      internal dependency) — without it, the extra installed but `test_midas_real_inference_sanity`
      still silently skipped with a different "module not found" message instead of running;
      added `timm` to the extra and verified the real MiDaS test then actually passes.

## Out of scope for now

Full detection -> tracking -> 3D -> planning -> safety "autonomous driving stack"
claims have been removed from this project's docs. If that's what you need, this
library can be one component (the tracking/prediction layer) of such a system, but
it is not one on its own.
