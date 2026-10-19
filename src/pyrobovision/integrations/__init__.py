"""Optional third-party adapters for PyRoboVision."""

from .alpamayo_adapter import (
    AlpamayoOutput,
    AlpamayoShapeError,
    AlpamayoTrajectory,
    AlpamayoTrajectoryAdapter,
    trajectory_errors,
)

__all__ = [
    "AlpamayoOutput",
    "AlpamayoShapeError",
    "AlpamayoTrajectory",
    "AlpamayoTrajectoryAdapter",
    "trajectory_errors",
]
