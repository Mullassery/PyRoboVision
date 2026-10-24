# Competitive intelligence in roadmap and upgrade work

**Policy date:** 2026-10-19

Competitive findings are an input to product decisions, not a one-off report.
Every pending roadmap item, new feature, and routine dependency upgrade should
check the repository's [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) before work
starts, and record the decision in the PR or changelog.

## Required check before building

1. **Read `AUDIT_FINDINGS.md`.** Find the relevant competitor row and the
   ranked next actions. If the finding is stale, update it with a dated source.
2. **Name the competitor and the user problem.** A feature is not justified by
   "competitor has it". State the workflow it unlocks, who needs it, and how
   often. Mark unverified demand as a hypothesis.
3. **Prefer reuse over new code.** Check whether an existing Mullassery repo
   already provides the capability (see `BUNDLES.md` in the profile repo and
   the bundle role file in this repo, when present). Typical reuse:
   - crawling/fetching/robots/SSRF → **MudFish**
   - architecture maps / PR diffs → **CodeAtlas**
   - data-quality contracts / validation → **StatGuardian**
   - token counting / cost tables → **PyTokenCalc**
   - dependency graph / advisories / SBOM → **PyDeepMesh**
   - segmentation / CLV / churn → **ClusterAudienceKit**
   - activation / lineage / PII → **PyReverseETL**
   - fleet state sync → **SyncFleet** (only after conflict resolution)
4. **Choose one disposition:** build, integrate, defer, or reject. Write it
   down. A rejection with a reason is a successful outcome.
5. **For routine upgrades**, check whether the upgrade closes a known
   competitor gap or breaks a differentiated capability. Run the tests that
   cover the affected path; do not merge a green build that silently changes a
   user-visible contract.

## PR / changelog checklist

- [ ] Competitor and source URL/date recorded (or "no direct competitor").
- [ ] User impact and frequency stated; unverified demand labelled.
- [ ] Reuse check done against sibling repos/bundles.
- [ ] Failure mode and rollback considered (especially auth, PII, cost).
- [ ] Docs/roadmap status updated honestly.
- [ ] Tests or a manual verification command included.

## Review cadence

- Refresh the competitor row when a release changes the capability.
- Re-check the full `AUDIT_FINDINGS.md` at least quarterly and before a major
  release.
- If a finding is wrong or stale, correct it in the same PR that relies on it.

## Non-goals

- Copying features for parity.
- Adding a dependency just because a competitor uses it.
- Treating star counts or download counts as proof of user need.

## OSS gap analysis — 2026-10-24

**What matters most here:** publish MOT benchmarks and fix occlusion re-association. The central identity-persistence claim is unmeasured, and a stale bbox weakens re-association; benchmark on MOT17/20 and use the Kalman projection.

Scope: PyRoboVision's tracking, occlusion, prediction, evaluation, and deployment surface was compared with four actively maintained Python tracking projects: BoxMOT, Norfair, OC-SORT, and Stone Soup. The comparison uses documented features and published benchmark/activity facts, not taglines.

### Comparable OSS projects

| Project | GitHub | License | Last activity | Why it is the comparison |
| --- | --- | --- | --- | --- |
| BoxMOT | https://github.com/mikel-brostrom/boxmot | AGPL-3.0 | v25.0.0, 2026-09-09 | Tracker zoo (ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, StrongSORT, BoostTrack) with ReID, MOT17 evaluation, tuning, and C++ backends; the default Python MOT choice. |
| Norfair | https://github.com/tryolabs/norfair | BSD-3-Clause | v2.3.0, 2025-04-30 | Lightweight detector-agnostic tracker with custom distances, ReID, camera-motion compensation, and published MOT17 scores; closest in spirit to our minimal core. |
| OC-SORT | https://github.com/noahcao/OC_SORT | MIT | Repo refactor noted 2026-03-19; no tagged release | Motion-only tracker whose observation-centric re-update is the reference for occlusion and non-linear-motion robustness. |
| Stone Soup | https://github.com/dstl/Stone-Soup | MIT | v1.9.1, 2026-06-24 | State-estimation/tracking framework with multiple filters, MHT/PDA association, CLEAR MOT and GOSPA metrics; comparison for prediction, uncertainty, and evaluation rigor. |

### Capability comparison

| Capability | Ours | BoxMOT | Norfair | OC-SORT | Stone Soup | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| Kalman + Hungarian MOT | Yes | Yes | Yes | Yes | Yes | Parity on the core loop. |
| Two-stage low-confidence association | Yes | Yes | No | Partial | No | ByteTrack-style concept is present; tuning maturity is lower. |
| Occlusion re-association | Partial | Yes | Yes | Yes | Yes | We predict through gaps but re-associate on the last detected bbox, not an observation-centric/predicted state. |
| Appearance ReID | No | Yes | Yes | No | No | No embedding hook, so long occlusions and crowds can swap IDs. |
| Camera-motion compensation | No | Yes | Yes | No | No | Moving robot platforms are unsupported; both leaders ship CMC. |
| Trajectory prediction with uncertainty | Yes | No | Partial | No | Yes | Ahead of MOT-only trackers; Stone Soup covers state prediction generally. |
| Learned trajectory model | Yes | No | No | No | No | Optional-torch GRU is unique among this set but small in scope. |
| Published MOT17/nuScenes scores | No | Yes | Yes | Yes | Partial | Credibility gap; competitors publish HOTA/MOTA/IDF1 tables. |
| Metrics harness (MOTA/IDF1/HOTA/OSPA) | No | Yes | Yes | Yes | Yes | No evaluation code or reproducible score script ships. |
| IMU/GPS sensor fusion | Yes | No | No | No | Yes | Useful breadth; not a tracker differentiator. |
| Depth/3D/LiDAR/occupancy utilities | Yes | No | Partial | No | No | We ship more perception plumbing than any tracker here. |
| C++/GPU/ONNX deployment | No | Yes | No | Partial | No | Large deployment gap; BoxMOT and OC-SORT have native/export paths. |
| NumPy/SciPy core, torch optional | Yes | No | Yes | No | Partial | Light install remains a real advantage. |
| Non-copyleft license | Yes | No | Yes | Yes | Yes | BoxMOT's AGPL is a blocker for some embedders. |

Nuance: "Partial" for our occlusion row means the lifecycle survives gaps, but `Track.get_bbox()` returns the last raw detection, so a fast object occluded long enough loses IoU overlap (documented in `CLAUDE.md`). OC-SORT's BYTE support is integration-level, not its primary two-stage design. Stone Soup is a general state-estimation framework, not an RGB MOT tracker, so its MOT score support is CLEAR MOT/GOSPA rather than a published MOT17 leaderboard entry.

### Gaps that matter (ranked)

1. **No published MOT benchmark results** — `docs/BENCHMARKS.md` explicitly says no formal benchmark has been run, while BoxMOT publishes 62-71 HOTA on MOT17 ablation, OC-SORT reports 63.2 HOTA, and Norfair publishes MOT17/MOT20 IDF1/MOTA. Users choosing an identity-persistence layer cannot verify our central claim; this hurts adoption first. Effort M: run MOT17 with public detections, publish HOTA/MOTA/IDF1 and the script.
2. **Occlusion re-association uses a stale bbox rather than predicted/observation-centric state** — our `Track.get_bbox()` returns the last detected box instead of the Kalman projection, so fast-moving objects can fail to re-associate after an occlusion; OC-SORT's ORU and BoxMOT's ReID path handle this. We have a weaker version, not an absent feature. Effort M.
3. **No appearance ReID hook** — BoxMOT and Norfair accept appearance embeddings, which is the main defense against ID swaps in crowds and long occlusions; our tracker is motion-only. Users with crowded scenes will leave. Effort M-L: optional embedding adapter behind an extra, not a bundled model.
4. **No camera-motion compensation** — Norfair and BoxMOT compensate for pan/tilt/zoom; our IoU association degrades when the camera moves, which is common on robots. Effort M.
5. **No evaluation/metrics harness** — Stone Soup ships CLEAR MOT and GOSPA, Norfair ships a motmetrics demo, and BoxMOT has `eval`/`tune`; we have unit tests only. This blocks both gap 1 and user self-evaluation. Effort S-M.
6. **Deployment gap** — BoxMOT offers native C++ with matching metrics and TensorRT/ONNX paths; OC-SORT supports C++/ONNX/TensorRT; Norfair has a ROS package. We are Python CPU only. Effort L; defer unless a robotics deployment user funds it.

### Where we are differentiated

- **Zero mandatory deep-learning runtime.** NumPy + SciPy only; torch is confined to optional depth/ONNX/learned-prediction paths. BoxMOT and OC-SORT are torch-first.
- **Apache-2.0, not AGPL.** BoxMOT's AGPL-3.0 is disqualifying for many closed robotics products; our license is embeddable.
- **Tracking plus forecasting, fusion, and 3D utilities in one package.** CV/CA + learned prediction with uncertainty, IMU/GPS fusion, depth, 3D boxes, LiDAR, and BEV occupancy are not shipped together by any competitor here.
- **Explicit Alpamayo output adapter.** CPU-side parsing of 64-waypoint/10 Hz trajectories and reasoning is a boundary none of these projects provide.
- **Honest capability labelling.** The README and roadmap separate stable, placeholder, and external features, which is rare in this comparison set.

### Disposition

- **Build:** MOT17 evaluation harness and published HOTA/MOTA/IDF1 table; predicted-state occlusion re-association; a small metrics module usable without benchmark data.
- **Integrate/reuse:** optional ReID embedding adapter (reuse permissive embedding models rather than training one); camera-motion compensation based on the Norfair (BSD) approach; keep the existing Supervision adapter as the detector boundary.
- **Defer/reject:** C++/GPU tracker, bundled detector, foundation models, and benchmark-chasing for its own sake. Reject copying AGPL BoxMOT code.

### Sources

- https://github.com/mikel-brostrom/boxmot (accessed 2026-10-24)
- https://pypi.org/project/boxmot (v25.0.0, 2026-09-09)
- https://github.com/tryolabs/norfair (accessed 2026-10-24)
- https://github.com/tryolabs/norfair/releases (v2.3.0, 2025-04-30)
- https://github.com/noahcao/OC_SORT (accessed 2026-10-24)
- https://github.com/dstl/Stone-Soup (accessed 2026-10-24)
- https://github.com/dstl/Stone-Soup/releases (v1.9.1, 2026-06-24)
- Repo files inspected: `README.md`, `COMPETITIVE_INTELLIGENCE.md`, `AUDIT_FINDINGS.md`, `ROADMAP_HONEST.md`, `pyproject.toml`, `CLAUDE.md`, `docs/BENCHMARKS.md`.
