"""Optional adapter for NVIDIA Alpamayo reasoning-VLA outputs.

Alpamayo is an open reasoning vision-language-action model for autonomous
driving. Its inference API returns ego trajectories and Chain-of-Causation
(CoC) reasoning traces. This module converts those outputs into plain NumPy
structures that PyRoboVision can consume; it does **not** import ``torch``,
``alpamayo_r1``, or download model weights, and it does not bundle or fork the
upstream code.

The model itself is an optional external dependency: Alpamayo 1 weights are
research/non-commercial, and inference requires an NVIDIA GPU with at least
24 GB VRAM. Keep this adapter on the boundary so the base package remains
CPU-only and dependency-light.

Upstream output shapes (Alpamayo 1)::

    pred_xyz: [batch, num_traj_sets, num_traj_samples, 64, 3]
    pred_rot: [batch, num_traj_sets, num_traj_samples, 64, 3, 3]
    extra["cot"]: reasoning traces (nested, one per trajectory sample)

The adapter also accepts the batch/set dimensions omitted, which makes it
straightforward to test with fixtures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_HORIZON = 64
DEFAULT_DT = 0.1


class AlpamayoShapeError(ValueError):
    """Raised when an Alpamayo output has an unsupported shape."""


def _as_numpy(value: Any, *, name: str) -> np.ndarray | None:
    """Convert tensors/arrays/lists to float64 NumPy without importing torch."""
    if value is None:
        return None

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AlpamayoShapeError(
            f"{name} cannot be converted to a numeric array"
        ) from exc


def _normalize_trajectories(
    value: Any,
    *,
    batch_index: int,
    coordinate_dims: tuple[int, ...] = (3,),
) -> np.ndarray:
    """Return trajectories as ``(num_samples, horizon, D)``."""
    array = _as_numpy(value, name="pred_xyz")
    if array is None:
        raise AlpamayoShapeError("pred_xyz is required")

    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        pass  # (samples, horizon, 3)
    elif array.ndim == 4:
        # (batch, samples, horizon, 3)
        if not 0 <= batch_index < array.shape[0]:
            raise AlpamayoShapeError(
                f"batch_index {batch_index} is out of range for batch size {array.shape[0]}"
            )
        array = array[batch_index]
    elif array.ndim == 5:
        # (batch, trajectory_sets, samples, horizon, 3)
        if not 0 <= batch_index < array.shape[0]:
            raise AlpamayoShapeError(
                f"batch_index {batch_index} is out of range for batch size {array.shape[0]}"
            )
        batch = array[batch_index]
        array = batch.reshape((-1,) + batch.shape[-2:])
    else:
        raise AlpamayoShapeError(
            "pred_xyz must have shape (T,3), (S,T,3), (B,S,T,3), or "
            f"(B,sets,S,T,3); got {array.shape}"
        )

    if array.ndim != 3 or array.shape[-1] not in coordinate_dims:
        raise AlpamayoShapeError(
            f"pred_xyz must resolve to (samples, horizon, {coordinate_dims}); "
            f"got {array.shape}"
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise AlpamayoShapeError(
            "pred_xyz must contain at least one sample and one waypoint"
        )
    if not np.all(np.isfinite(array)):
        raise AlpamayoShapeError("pred_xyz contains non-finite values")
    return array


def _normalize_rotations(value: Any, *, batch_index: int) -> np.ndarray | None:
    """Return rotations as ``(num_samples, horizon, 3, 3)`` when supplied."""
    array = _as_numpy(value, name="pred_rot")
    if array is None:
        return None

    if array.ndim == 3 and array.shape[-1] == 9:
        array = array.reshape(array.shape[0], array.shape[1], 3, 3)
    elif array.ndim == 3 and array.shape[-2:] == (3, 3):
        array = array[None, ...]
    elif array.ndim == 4:
        if array.shape[-2:] != (3, 3):
            raise AlpamayoShapeError(f"pred_rot must end in (3,3); got {array.shape}")
    elif array.ndim == 5 or array.ndim == 6:
        if array.shape[-2:] != (3, 3):
            raise AlpamayoShapeError(f"pred_rot must end in (3,3); got {array.shape}")
        if not 0 <= batch_index < array.shape[0]:
            raise AlpamayoShapeError(
                f"batch_index {batch_index} is out of range for batch size {array.shape[0]}"
            )
        batch = array[batch_index]
        array = batch.reshape((-1,) + batch.shape[-3:])
    else:
        raise AlpamayoShapeError(
            "pred_rot must have shape (T,3,3), (S,T,3,3), (B,S,T,3,3), "
            f"or (B,sets,S,T,3,3); got {array.shape}"
        )

    if array.ndim != 4 or array.shape[-2:] != (3, 3):
        raise AlpamayoShapeError(
            f"pred_rot must resolve to (samples, horizon, 3, 3); got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise AlpamayoShapeError("pred_rot contains non-finite values")
    return array


def _flatten_reasoning(value: Any) -> list[str]:
    """Flatten nested reasoning traces into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        for key in ("text", "content", "reasoning", "cot", "value"):
            if key in value:
                return _flatten_reasoning(value[key])
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        traces: list[str] = []
        for item in value:
            traces.extend(_flatten_reasoning(item))
        return traces
    return []


def _extract_reasoning(
    extra: Any, *, num_samples: int
) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    """Return all traces and a per-sample assignment."""
    if extra is None:
        return (), (None,) * num_samples

    source = extra
    if isinstance(extra, Mapping):
        source = None
        for key in ("cot", "reasoning", "reasoning_traces", "chain_of_causation"):
            if key in extra:
                source = extra[key]
                break

    traces = tuple(_flatten_reasoning(source))
    if not traces:
        return (), (None,) * num_samples

    if len(traces) == num_samples:
        return traces, traces
    if len(traces) < num_samples:
        assigned = list(traces) + [None] * (num_samples - len(traces))
        return traces, tuple(assigned)
    return traces, tuple(traces[:num_samples])


@dataclass(frozen=True)
class AlpamayoTrajectory:
    """One Alpamayo trajectory sample.

    Attributes:
        xyz: Ego-frame waypoints with shape ``(horizon, 3)`` in metres.
        rotation: Optional rotation matrices with shape ``(horizon, 3, 3)``.
        reasoning: Chain-of-Causation text for this sample, if available.
        sample_index: Index of this sample within the inference result.
    """

    xyz: np.ndarray
    rotation: np.ndarray | None
    reasoning: str | None
    sample_index: int


@dataclass(frozen=True)
class AlpamayoOutput:
    """Normalized Alpamayo inference output."""

    trajectories: tuple[AlpamayoTrajectory, ...]
    reasoning: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return len(self.trajectories)

    def best(self) -> AlpamayoTrajectory:
        """Return the first trajectory sample (Alpamayo's default sample)."""
        return self.trajectories[0]

    def positions_2d(self) -> tuple[np.ndarray, ...]:
        """Return ``(horizon, 2)`` arrays for every sample."""
        return tuple(trajectory.xyz[:, :2].copy() for trajectory in self.trajectories)


class AlpamayoTrajectoryAdapter:
    """Convert Alpamayo inference outputs into PyRoboVision-friendly arrays.

    Example::

        adapter = AlpamayoTrajectoryAdapter()
        output = adapter.parse(pred_xyz, pred_rot, extra)
        for sample in output.trajectories:
            print(sample.reasoning, sample.xyz[:5])

    The adapter is deliberately model-agnostic: callers can pass outputs from
    the upstream ``AlpamayoR1`` model, a saved fixture, or a compatible
    inference service.
    """

    def __init__(
        self,
        *,
        horizon: int | None = DEFAULT_HORIZON,
        dt: float = DEFAULT_DT,
        batch_index: int = 0,
    ) -> None:
        if horizon is not None and horizon < 1:
            raise ValueError("horizon must be positive or None")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.horizon = horizon
        self.dt = dt
        self.batch_index = batch_index

    def parse(
        self,
        pred_xyz: Any,
        pred_rot: Any = None,
        extra: Any = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> AlpamayoOutput:
        """Normalize an Alpamayo inference result.

        Args:
            pred_xyz: Trajectory tensor/array. Accepted shapes are
                ``(T,3)``, ``(S,T,3)``, ``(B,S,T,3)``, and
                ``(B,sets,S,T,3)``.
            pred_rot: Optional rotations in a matching leading shape, with a
                final ``(3,3)`` or flattened ``(9,)`` dimension.
            extra: Optional mapping containing ``cot``/``reasoning`` traces.
            metadata: Optional metadata copied into the output.

        Raises:
            AlpamayoShapeError: If a shape is unsupported or non-finite.
            ValueError: If the waypoint count does not match ``horizon``.
        """
        trajectories = _normalize_trajectories(pred_xyz, batch_index=self.batch_index)
        rotations = _normalize_rotations(pred_rot, batch_index=self.batch_index)

        if rotations is not None and rotations.shape[:2] != trajectories.shape[:2]:
            raise AlpamayoShapeError(
                "pred_rot and pred_xyz must have matching sample and horizon dimensions; "
                f"got {rotations.shape[:2]} and {trajectories.shape[:2]}"
            )
        if self.horizon is not None and trajectories.shape[1] != self.horizon:
            raise ValueError(
                f"expected {self.horizon} waypoints, got {trajectories.shape[1]}"
            )

        all_reasoning, per_sample = _extract_reasoning(
            extra, num_samples=trajectories.shape[0]
        )
        samples = tuple(
            AlpamayoTrajectory(
                xyz=trajectories[index].copy(),
                rotation=None if rotations is None else rotations[index].copy(),
                reasoning=per_sample[index],
                sample_index=index,
            )
            for index in range(trajectories.shape[0])
        )

        output_metadata: dict[str, Any] = {
            "source": "alpamayo",
            "horizon": int(trajectories.shape[1]),
            "dt": self.dt,
            "num_samples": int(trajectories.shape[0]),
        }
        if metadata:
            output_metadata.update(metadata)

        return AlpamayoOutput(
            trajectories=samples,
            reasoning=all_reasoning,
            metadata=output_metadata,
        )

    def select_best(
        self,
        output: AlpamayoOutput,
        ground_truth: Any,
    ) -> tuple[AlpamayoTrajectory, dict[str, Any]]:
        """Select the sample with the lowest minADE against ground truth."""
        metrics = trajectory_errors(
            np.stack([sample.xyz for sample in output.trajectories]),
            ground_truth,
        )
        best_index = int(metrics["best_sample"])
        return output.trajectories[best_index], metrics


def trajectory_errors(predicted: Any, ground_truth: Any) -> dict[str, Any]:
    """Compute ADE/FDE for one or more predicted trajectories.

    Args:
        predicted: ``(T,D)`` or ``(S,T,D)`` predicted waypoints.
        ground_truth: ``(T,D)`` reference waypoints. ``D`` must be 2 or 3.

    Returns:
        A dictionary with ``min_ade``, ``min_fde``, ``best_sample``,
        per-sample ``ade``/``fde`` lists, and the per-timestep error matrix.

    When ground truth is 2D, predicted z coordinates are ignored. Non-finite
    inputs and unsupported shapes raise ``AlpamayoShapeError``.
    """
    pred = _normalize_trajectories(predicted, batch_index=0, coordinate_dims=(2, 3))
    target = _as_numpy(ground_truth, name="ground_truth")
    if target is None:
        raise AlpamayoShapeError("ground_truth is required")
    if target.ndim != 2 or target.shape[1] not in (2, 3):
        raise AlpamayoShapeError(
            f"ground_truth must have shape (T,2) or (T,3); got {target.shape}"
        )
    if target.shape[1] > pred.shape[-1]:
        raise AlpamayoShapeError(
            "ground_truth has more dimensions than predicted trajectories"
        )
    if not np.all(np.isfinite(target)):
        raise AlpamayoShapeError("ground_truth contains non-finite values")

    horizon = min(pred.shape[1], target.shape[0])
    if horizon < 1:
        raise AlpamayoShapeError("trajectories must share at least one waypoint")

    pred_xyz = pred[:, :horizon, : target.shape[1]]
    target_xyz = target[:horizon]
    per_timestep = np.linalg.norm(pred_xyz - target_xyz[None, :, :], axis=-1)
    per_sample_ade = per_timestep.mean(axis=1)
    per_sample_fde = per_timestep[:, -1]
    best_sample = int(np.argmin(per_sample_ade))

    return {
        "min_ade": float(per_sample_ade[best_sample]),
        "min_fde": float(per_sample_fde[best_sample]),
        "best_sample": best_sample,
        "ade": [float(value) for value in per_sample_ade],
        "fde": [float(value) for value in per_sample_fde],
        "per_timestep": per_timestep.tolist(),
        "horizon": int(horizon),
    }


__all__ = [
    "DEFAULT_DT",
    "DEFAULT_HORIZON",
    "AlpamayoOutput",
    "AlpamayoShapeError",
    "AlpamayoTrajectory",
    "AlpamayoTrajectoryAdapter",
    "trajectory_errors",
]
