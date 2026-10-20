# Alpamayo Workflow

Status: **existing boundary adapter; Alpamayo runtime and weights are external.**

PyRoboVision converts Alpamayo trajectory outputs into plain NumPy objects without importing `torch`, `alpamayo_r1`, or model code. Alpamayo inference runs in a separate Linux/NVIDIA environment.

## Contract

Alpamayo 1.5 and 2 preserve the Alpamayo 1 trajectory contract used by the adapter:

```text
pred_xyz: (T, 3) | (S, T, 3) | (B, S, T, 3) | (B, sets, S, T, 3)
pred_rot: matching leading shape + (3, 3) or (9,)
T = 64 waypoints
dt = 0.1 s
extra["cot"] / extra["reasoning"] = Chain-of-Causation text
```

Alpamayo 1.5 adds navigation guidance, flexible camera counts, and VQA. Alpamayo 2 Super is a larger multi-task model. Those optional outputs are not parsed by the current adapter.

## Install and run

Use the upstream Alpamayo environment. Do not install Alpamayo into the PyRoboVision base environment.

```bash
# Illustrative upstream flow; follow the Alpamayo release you are using.
git clone <ALPAMAYO_REPO>
cd <ALPAMAYO_REPO>
uv venv
source .venv/bin/activate
uv pip install -e .
```

Alpamayo 1.5 inference requires a Linux host with an NVIDIA GPU and CUDA 12.x; the model card states roughly 24 GB VRAM for one sample. Alpamayo 2 Super is a 34B model and needs substantially more.

## Parse in PyRoboVision

```python
from pyrobovision.integrations import AlpamayoTrajectoryAdapter

adapter = AlpamayoTrajectoryAdapter(horizon=64, dt=0.1)
output = adapter.parse(
    pred_xyz,
    pred_rot,
    extra,
    metadata={"model_version": "alpamayo-1.5"},
)

for trajectory in output.trajectories:
    print(trajectory.sample_index, trajectory.xyz.shape, trajectory.reasoning)

best = output.best()
positions_2d = output.positions_2d()
```

Compute errors:

```python
from pyrobovision.integrations import trajectory_errors

metrics = trajectory_errors(pred_xyz, ground_truth_xyz)
```

## Synthetic version fixtures

CPU-only contract fixtures live in `tests/fixtures/alpamayo/`:

- `alpamayo_1_5_synthetic.json`
- `alpamayo_2_synthetic.json`

They contain 64 waypoints at 10 Hz, two trajectory samples, Chain-of-Causation text, and explicit `model_version` metadata. They are synthetic: they are not captured model output and contain no weights. Use them to test shape handling and provenance plumbing, not model quality.

```python
import json
from pathlib import Path

payload = json.loads(
    Path("tests/fixtures/alpamayo/alpamayo_2_synthetic.json").read_text()
)
output = adapter.parse(
    payload["pred_xyz"],
    extra={"cot": payload["cot"]},
    metadata={"model_version": payload["model_version"]},
)
```

## Live artifact check

When a real inference artifact is available, run the opt-in adapter check:

```bash
ALPAMAYO_ARTIFACT=/data/alpamayo/run.json \
  PYTHONPATH=src python3 -m pytest tests/test_alpamayo_live.py
```

The test accepts `.json` and `.npz` artifacts and asserts the 64-waypoint/10 Hz contract. It skips unless `ALPAMAYO_ARTIFACT` is set and never loads model weights.

## Provenance

- Alpamayo source: Apache-2.0.
- Alpamayo weights: OpenMDW-1.1; model cards describe non-commercial use with commercial licensing available on request.
- Do not bundle or redistribute weights.
- Record `model_version`, checkpoint ID, camera set, and inference commit in artifact metadata.

## Limitations

- No model loading, inference, training, or TensorRT export.
- No VQA, grounding, or meta-action parsing.
- Shape validation does not prove semantic correctness.
- GPU inference and fixture replay against live Alpamayo outputs are not run in this environment.

The full install, licensing, and validation workflow is in
`reports/NVIDIA_INTEGRATION_WORKFLOW.md`.
