"""Opt-in parsing check against a real Alpamayo artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pyrobovision.integrations import AlpamayoTrajectoryAdapter

ALPAMAYO_ARTIFACT = os.environ.get("ALPAMAYO_ARTIFACT")


def _load_artifact(path: Path) -> dict:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    return json.loads(path.read_text())


@pytest.mark.skipif(
    not ALPAMAYO_ARTIFACT, reason="set ALPAMAYO_ARTIFACT to a real run file"
)
def test_live_alpamayo_artifact():
    path = Path(ALPAMAYO_ARTIFACT)
    assert path.is_file()
    payload = _load_artifact(path)

    output = AlpamayoTrajectoryAdapter().parse(
        payload["pred_xyz"],
        payload.get("pred_rot"),
        extra={"cot": payload.get("cot")},
        metadata={"model_version": payload.get("model_version")},
    )

    assert output.num_samples >= 1
    assert output.metadata["horizon"] == 64
    assert all(sample.xyz.shape == (64, 3) for sample in output.trajectories)
