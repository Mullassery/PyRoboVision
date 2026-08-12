# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## What this project actually is

PyRoboVision is a pure Python (NumPy + SciPy only) **multi-object tracking and
trajectory prediction** library: Kalman filter state estimation + Hungarian algorithm
association, with real occlusion handling and constant velocity/acceleration
trajectory prediction. That core is genuinely implemented and tested (277 tests,
~89% coverage).

Around it are supporting utilities of varying maturity — see the table in
[README.md](./README.md#whats-real-vs-placeholder) for the honest real/placeholder
breakdown before you assume something works. In particular:

- `perception/depth.py` has two backends: `model="heuristic"` (default, no extra
  deps, NOT a real depth model — a Sobel-edge placeholder) and `model="midas"` (real
  pretrained MiDaS_small via `torch.hub`, requires `pip install "pyrobovision[depth]"`
  and, on first use, a network-dependent one-time weight download).
- There is **no bundled object detector**, no GPU inference pipeline, and no
  foundation-model integration. A previous version of this repo had an "MCP 2.0"
  module and CLI/server "workflow" files that looked like they did video
  processing / foundation-model inference but only returned hardcoded fake values
  regardless of input — those were deleted, not fixed, because there was nothing
  real underneath to fix.

If you're asked to "wire up" or "use" a class that isn't in the table above,
`grep -rn "class <Name>" src/` first — this codebase has previously had docs
reference classes (`AutonomousStack`, `Vision`, `CylindricalStitcher`,
`MultiModalFusion`, `get_waymo_layout`, SAM3/CLIP/Grounding DINO wrappers) that never
existed anywhere in `src/`. Don't assume prose in a doc file describes real code —
verify against `src/pyrobovision/`.

## Code structure (`src/pyrobovision/`)

- `tracking/` — the core. `kalman_filter.py` (constant-velocity KF), `association.py`
  (IoU cost matrix + `scipy.optimize.linear_sum_assignment`), `mot.py` (`MOTTracker`,
  `Track`, `Detection`: birth/confirm/occlude/death lifecycle).
- `prediction/` — `trajectory.py` (`ConstantVelocityModel`, `ConstantAccelerationModel`,
  `TrajectoryPredictor`), `uncertainty.py`.
- `perception/` — `depth.py` (real MiDaS + heuristic placeholder, see above),
  `bbox_3d.py` (2D bbox + depth -> 3D box), `lidar.py` (point cloud filtering/clustering),
  `occupancy.py` (BEV occupancy grids).
- `behavior/`, `intent/` — rule-based (not learned) motion/intent classification.
- `learning/` — imitation learning, behavior cloning, safety-constraint validation,
  a small training loop. Real but minimal; not benchmarked against production RL code.
- `fusion/` — `sensor_fusion.py` (Kalman-based IMU/GPS fusion), `optimization.py`
  (`ModelOptimizer`: thin wrapper around `torch.onnx.export`, lazy-imports torch so
  it doesn't force the dependency on everyone).

**Key design**: the core (`tracking/`, `prediction/`) only ever imports NumPy/SciPy.
Keep it that way — don't add a hard `torch`/`transformers` import to anything outside
`perception/depth.py`'s midas path and `fusion/optimization.py`'s ONNX path without a
strong reason; those two already lazy-import torch specifically so `pip install
pyrobovision` stays lightweight.

## Build & Test Commands

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"          # core dev deps only — no torch
pytest tests/ -v                 # 277 tests, ~89% coverage, all pass without torch
pytest tests/ -v --cov=pyrobovision --cov-report=term-missing

# Optional extras
pip install -e ".[dev,depth]"    # + torch, for DepthEstimator(model="midas")
pip install -e ".[dev,onnx]"     # + onnx/onnxruntime, for ModelOptimizer export
```

Format & lint: `black src/ tests/`, `isort src/ tests/`, `mypy src/pyrobovision/`,
`ruff check src/pyrobovision/`.

## Implementation notes worth knowing

- `MOTTracker.update()` always calls `predict()` on every existing track first (even
  ones with no matching detection this frame) — that's what makes occlusion
  prediction work. `Track.get_bbox()` for an occluded track returns the *last raw
  detected bbox*, not a Kalman-projected one, so re-association after occlusion
  relies on IoU overlap with where the object was last actually seen, not where it's
  predicted to be now. If you improve this, add a test in `tests/test_occlusion.py`
  for a fast-moving object occluded long enough that the stale bbox no longer
  overlaps the reappearing detection.
- `Track.is_confirmed` only ever goes `False -> True` (via `hit_streak >= 3` on
  `update()`); it never resets to `False` on missed frames, so a confirmed track
  stays "confirmed" (and reportable) through an occlusion gap up to `max_age`.
- `DepthEstimator`'s `midas` path returns *relative* inverse depth normalized into
  `[min_depth, max_depth]`, not metric depth — see the module docstring in
  `perception/depth.py` before treating its output as meters.

## License

Proprietary (same as PyRoboFrames) — Georgi Mammen Mullassery.
