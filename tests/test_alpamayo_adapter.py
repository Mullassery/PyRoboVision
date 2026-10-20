"""Alpamayo output adapter tests use synthetic fixtures; no model is loaded."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyrobovision.integrations.alpamayo_adapter import (
    AlpamayoShapeError,
    AlpamayoTrajectoryAdapter,
    trajectory_errors,
)


class _FakeTensor:
    """Minimal torch-like wrapper to verify the adapter never imports torch."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def _trajectories(num_samples=2, horizon=64):
    base = np.zeros((num_samples, horizon, 3), dtype=np.float64)
    base[:, :, 0] = np.arange(horizon) * 0.5
    base[:, :, 1] = np.arange(num_samples)[:, None] * 0.1
    return base


def test_parse_nested_alpamayo_output_and_reasoning():
    pred_xyz = _trajectories()[None, None, ...]
    pred_rot = np.broadcast_to(np.eye(3), (1, 1, 2, 64, 3, 3))
    extra = {"cot": [["Nudge left for cones.", "Yield to the cyclist."]]}

    output = AlpamayoTrajectoryAdapter().parse(pred_xyz, pred_rot, extra)

    assert output.num_samples == 2
    assert output.best().xyz.shape == (64, 3)
    assert output.trajectories[0].rotation.shape == (64, 3, 3)
    assert output.trajectories[0].reasoning == "Nudge left for cones."
    assert output.trajectories[1].reasoning == "Yield to the cyclist."
    assert output.metadata["horizon"] == 64


def test_parse_accepts_batch_without_set_dimension_and_torch_like_tensor():
    pred_xyz = _FakeTensor(_trajectories(num_samples=1)[None, ...])

    output = AlpamayoTrajectoryAdapter().parse(pred_xyz)

    assert output.num_samples == 1
    assert output.trajectories[0].rotation is None
    assert output.trajectories[0].xyz.dtype == np.float64


def test_reasoning_is_preserved_when_traces_do_not_map_one_to_one():
    pred_xyz = _trajectories(num_samples=2)
    extra = {"reasoning": "A single shared trace."}

    output = AlpamayoTrajectoryAdapter().parse(pred_xyz, extra=extra)

    assert output.reasoning == ("A single shared trace.",)
    assert output.trajectories[0].reasoning == "A single shared trace."
    assert output.trajectories[1].reasoning is None


def test_rejects_unsupported_shape_and_non_finite_values():
    adapter = AlpamayoTrajectoryAdapter()
    with pytest.raises(AlpamayoShapeError, match="pred_xyz"):
        adapter.parse(np.zeros((64, 4)))

    bad = _trajectories()
    bad[0, 0, 0] = np.nan
    with pytest.raises(AlpamayoShapeError, match="non-finite"):
        adapter.parse(bad)


def test_rejects_horizon_mismatch():
    with pytest.raises(ValueError, match="expected 64 waypoints"):
        AlpamayoTrajectoryAdapter().parse(np.zeros((10, 3)))


def test_trajectory_errors_and_best_selection():
    predicted = np.zeros((2, 3, 2))
    predicted[0] = [[1, 0], [2, 0], [3, 0]]
    predicted[1] = [[0, 0], [0, 0], [0, 0]]
    ground_truth = np.zeros((3, 2))

    metrics = trajectory_errors(predicted, ground_truth)

    assert metrics["best_sample"] == 1
    assert metrics["min_ade"] == 0.0
    assert metrics["min_fde"] == 0.0
    assert metrics["ade"][0] == pytest.approx(2.0)

    predicted_3d = np.concatenate(
        [predicted, np.zeros((2, 3, 1), dtype=np.float64)], axis=-1
    )
    adapter = AlpamayoTrajectoryAdapter(horizon=3)
    output = adapter.parse(predicted_3d)
    best, selected = adapter.select_best(output, ground_truth)
    assert output.num_samples == 2
    assert best.sample_index == 1
    assert selected["best_sample"] == 1


def test_positions_2d_returns_independent_copies():
    output = AlpamayoTrajectoryAdapter().parse(_trajectories(num_samples=1))
    positions = output.positions_2d()[0]
    positions[0, 0] = 999.0

    assert output.best().xyz[0, 0] != 999.0
    assert positions.shape == (64, 2)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "alpamayo"


@pytest.mark.parametrize(
    ("filename", "model_version"),
    [
        ("alpamayo_1_5_synthetic.json", "alpamayo-1.5"),
        ("alpamayo_2_synthetic.json", "alpamayo-2"),
    ],
)
def test_synthetic_model_version_fixtures_preserve_contract(
    filename, model_version
):
    payload = json.loads((FIXTURE_DIR / filename).read_text())

    output = AlpamayoTrajectoryAdapter().parse(
        payload["pred_xyz"],
        extra={"cot": payload["cot"]},
        metadata={
            "model_version": payload["model_version"],
            "model": payload["model"],
        },
    )

    assert payload["fixture"] == "synthetic"
    assert payload["contract"] == {
        "horizon": 64,
        "dt": 0.1,
        "coordinate_frame": "ego",
    }
    assert output.num_samples == 2
    assert output.best().xyz.shape == (64, 3)
    assert output.metadata["horizon"] == 64
    assert output.metadata["dt"] == 0.1
    assert output.metadata["model_version"] == model_version
    assert output.trajectories[0].reasoning == payload["cot"][0]
    assert output.trajectories[1].reasoning == payload["cot"][1]
