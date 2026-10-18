# Audit findings — 2026-10-19

This file records the repository-specific findings from the portfolio audit
conducted on 2026-10-19. It is documentation only; no code or release artifacts
were changed in this pass unless explicitly noted under "Implementation status".

Sources: the repository's own README/source/workflows, GitHub metadata and
Actions runs, PyPI metadata, and official competitor documentation. Claims are
marked as verified, inferred, or open questions. "Absent" means not found in the
inspected repository/docs, not proof that it cannot exist.

## What it is

Multi-object tracking, trajectory prediction, depth estimation, 3D perception utilities, sensor fusion, and learning helpers for robotics.

## Verified strengths

- Kalman + Hungarian tracking; occlusion/lifecycle; CV/CA prediction; MiDaS optional; 3D box/LiDAR/BEV; IMU/GPS fusion; ONNX; 278 tests/84% coverage.
- Good utility breadth for a small package.

## Verified gaps and user-facing risks

- No object detector; CPU only; heuristic depth default is a Sobel placeholder; no foundation models.
- No MOT benchmark; standard algorithms with weak differentiation.

## Direct competitors

- ByteTrack/OC-SORT/BoT-SORT, Ultralytics, Norfair, OpenPCDet.

## Ranked next actions

- P3: integrate a detector + GPU path (L).
- P3: remove/label the heuristic depth default (S).

## Do not build

- Foundation models/CLIP/SAM; out of scope and costly.

## Evidence gaps / validation needed

- No MOT17/nuScenes results; no accuracy comparison.

## Recommended disposition

Freeze unless a detector integration is funded.

## Implementation status

No code changes in this pass.
