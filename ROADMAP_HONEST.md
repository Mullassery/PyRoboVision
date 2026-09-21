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

## Technical Debt

Assessed and re-verified during the 2026-09 OSS maturity pass. Two config bugs were
fixed directly (safe, mechanical); a follow-up quick-fix pass then resolved the two
bare-`except:` clauses and the dev-tool CVE ceilings (see below); everything else
remaining is real, unaddressed debt left for a dedicated follow-up rather than fixed
here.

### Fixed in this pass (safe, mechanical)

- **`pyproject.toml`** — `[tool.isort]` had `multi_line_mode = 3`, which is not a
  valid isort setting (the real option is `multi_line_output`, already implied by
  `profile = "black"`). This wasn't a lint *warning* — it made `isort` hard-crash
  with `UnsupportedSettings` on every invocation, so `isort --check-only .` and
  `make lint` never actually ran. Removed the invalid key; `isort` now runs (and
  reports real formatting diffs — see below).
- **`Makefile`** — `test-cov` ran `pytest -v --cov=pyrobovision` and `lint` ran
  `mypy pyrobovision`, both using the package name instead of the `src/` path. This
  is the exact same silent-failure class already documented in
  `pyproject.toml`'s own `[tool.pytest.ini_options]` comment (`--cov=pyrobovision`
  reports "no data collected" under this `src/` layout instead of erroring) — the
  fix was applied to `pyproject.toml`'s `addopts` previously but never mirrored into
  the `Makefile`. Now both use `src/pyrobovision`.
- **`.github/workflows/tests.yml`** — `actions/setup-python@v4` (flagged by
  `actionlint` as running on a deprecated Node runtime) bumped to `@v5` in both jobs.
- **`.github/CI_ERRORS.md`** — deleted. It was generic boilerplate pointing at
  `../../CI_ERRORS_REPORT.md` and `.github/TROUBLESHOOTING.md`, neither of which
  exists anywhere in this repo; it contained no repo-specific content.
- **`CONTRIBUTING.md`** — "By contributing, you agree your contributions are
  licensed under the Proprietary License" was stale from before the Apache-2.0
  relicense (`f2d77e7`); corrected to Apache License 2.0.
- **`CLAUDE.md`** — said "Proprietary (same as PyRoboFrames)" under License (stale,
  same relicense miss as above) and "277 tests, ~89% coverage" (stale, superseded by
  the 285-test / 84%-coverage numbers already correct in this file's own
  `ROADMAP_HONEST.md`). Both corrected.
- **`README.md`** — test/coverage badges and prose said "276 passing / 1 skipped,
  87% coverage, 1895 statements / 244 missed"; re-measured for this pass and it's
  actually **278 passed / 7 skipped, 84% coverage, 1982 statements / 317 missed**
  (`pytest tests/ -q` on Python 3.11, no `torch` installed). The old numbers predate
  the `LearnedTrajectoryModel` tests being added and were never updated.

### Not fixed — needs a dedicated follow-up session

- **Formatting drift, not enforced anywhere**: `black --check src/ tests/` reports
  **32 of 48 files** would be reformatted (e.g.
  `src/pyrobovision/tracking/mot.py`, `src/pyrobovision/perception/depth.py`,
  `src/pyrobovision/perception/{lidar,occupancy}.py`,
  `src/pyrobovision/prediction/trajectory.py`, and most of `tests/`). `isort
  --check-only` (now that it actually runs — see above) fails on **19 of 21** test
  files. Nothing in CI checks formatting; `.pre-commit-config.yaml` has black/isort/
  ruff hooks but they're opt-in (`pre-commit install`), not enforced server-side.
  Fixing this is a single mechanical `make fmt` run, but it touches ~35 files and
  should be its own PR, not bundled into a docs pass.
- **`ruff check src/pyrobovision`: 221 findings (168 auto-fixable)** — was 223
  before the two `E722` bare-excepts below were fixed. Breakdown:
  105× `UP006`/40× `UP035` (old `typing.Dict`/`typing.List`/`typing.Tuple` instead
  of builtin generics — the package requires Python >=3.10 but was never modernized
  to PEP 585/604 syntax), 25× `I001` unsorted imports, 16× `F401` unused imports
  (e.g. `MotionClassification` imported but unused in
  `src/pyrobovision/behavior/analyzer.py:4`, `typing.List` unused in
  `src/pyrobovision/behavior/patterns.py:3`, `typing.Tuple` unused in
  `src/pyrobovision/fusion/optimization.py:2`), 15× `UP045` (`Optional[X]` instead of
  `X | None`), 3× `F841` unused variables. ~~2× `E722` bare `except:` clauses~~ —
  **fixed** in the 2026-09 quick-fix follow-up: `src/pyrobovision/perception/bbox_3d.py:147`
  and `src/pyrobovision/perception/lidar.py:202` now catch `np.linalg.LinAlgError`
  specifically (the only exception `np.linalg.eig` actually raises, on
  non-convergence) instead of every exception including
  `KeyboardInterrupt`/`SystemExit`; `ruff check src/pyrobovision --select E722` is
  now clean and the full test suite still passes (278 passed / 7 skipped).
  Everything else in this ruff bullet remains style/modernization debt, unaddressed.
- **`mypy src/pyrobovision`: 72 errors in 15 files.** Two categories: (1) missing
  variable annotations under `disallow_untyped_defs = false` still triggering
  `var-annotated` errors for list/dict attributes initialized as `[]`/`{}` in
  `src/pyrobovision/learning/policy.py:29-33,131`,
  `src/pyrobovision/learning/imitation.py:162-165`,
  `src/pyrobovision/learning/behavior_cloning.py:35-40,90-91`, and
  `src/pyrobovision/fusion/sensor_fusion.py:60`; (2) `import-not-found` for
  `onnx`/`onnxruntime`/`torch`/`tensorrt` in
  `src/pyrobovision/fusion/optimization.py:25-26,34,66,108` — expected since those
  are optional-extra deps not installed in the base dev environment, but
  `pyproject.toml`'s `[tool.mypy]` has no per-module override
  (`ignore_missing_imports`) for them, so a plain `mypy src/pyrobovision` run always
  shows these as errors regardless of who's running it or why.
- ~~**Known vulnerabilities in pinned dev-tool ceilings**~~ — **fixed** in the
  2026-09 quick-fix follow-up: `pyproject.toml`'s dev ceilings were
  `black>=23.0,<26.0` (line 54) and `pytest>=7.4,<9.0` (line 52), which blocked
  the versions carrying fixes for `PYSEC-2026-2121`/`PYSEC-2026-2120` (black,
  fixed in `26.3.0`/`26.3.1`) and `PYSEC-2026-1845` (pytest, fixed in `9.0.3`).
  Bumped to `black>=23.0,<27.0` and `pytest>=7.4,<10.0`; installs now resolve to
  `black==26.5.1` / `pytest==9.1.1`. Verified: `pip-audit` now reports zero known
  vulnerabilities (previously 3 advisories across the two packages); full test
  suite still 278 passed / 7 skipped; `black --check src/ tests/` still flags
  the same 32/48 files as before the bump — the newer black did not introduce
  additional formatting drift. Both are dev-only tooling (not shipped to anyone
  who does `pip install pyrobovision`). The `dependency-audit` CI job
  (`pip-audit`, `continue-on-error: true`) added in the original OSS pass stays
  in place as an informational, non-blocking check.
- **No CI enforcement of lint/format/types at all.** `.github/workflows/tests.yml`
  only runs `pytest`. `Makefile`'s `lint` target and `CONTRIBUTING.md`'s "Before
  opening a PR" section both tell contributors to run black/isort/ruff/mypy, but
  nothing checks it server-side. Deliberately not added as a blocking CI job in this
  pass, because turning it on today would immediately fail on the pre-existing drift
  above — the real fix is: run `make fmt`, fix the mypy `var-annotated` errors, add
  mypy overrides for the optional-extra imports, *then* add the CI gate. (The two
  bare-excepts that used to be on this list are already fixed — see above.)
