"""Sparse/masked data support for handling sensor failures and missing values.

Real robots have sensor outages, network dropouts, and calibration failures.
This module handles gracefully.

```python
from pyroboframes.masking import MaskedDataFrame, interpolate_missing

# Create masked view
mdf = MaskedDataFrame(df, mask_key="sensor_valid")  # Boolean column indicating valid data
print(mdf.coverage_report())  # Coverage % per sensor

# Interpolate missing values
df_filled = interpolate_missing(df, method="forward_fill", columns=["imu_accel"])
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa

    from .dataframe import RoboticsDataFrame


class MaskedDataFrame:
    """Wrapper for DataFrames with optional/masked sensor data.

    Tracks which columns have valid data at each timestep.
    """

    def __init__(
        self,
        dataframe: RoboticsDataFrame,
        mask_key: str | None = None,
        sentinel_value: float = np.nan,
    ):
        """Initialize masked view.

        Args:
            dataframe: RoboticsDataFrame
            mask_key: Column name with boolean validity mask (default: auto-detect via NaN)
            sentinel_value: Value indicating missing data
        """
        self._df = dataframe
        self.mask_key = mask_key
        self.sentinel = sentinel_value
        self._mask_cache = {}

    def coverage_by_column(self, episode_index: int = 0) -> dict[str, float]:
        """Get valid data coverage % for each column in an episode.

        Args:
            episode_index: Episode to analyze

        Returns:
            Dict mapping column_name → coverage (0-1)
        """
        ep_slice = self._df.slice(episode_index=episode_index)
        ep_table = ep_slice.to_pyarrow()

        coverage = {}
        for col_name in ep_table.column_names:
            col = ep_table[col_name].combine_chunks().to_numpy()
            if col.dtype == object:
                # Nested (list) column; check for None
                valid = np.array([v is not None for v in col])
            elif np.issubdtype(col.dtype, np.floating):
                valid = ~np.isnan(col)
            else:
                valid = np.ones(len(col), dtype=bool)

            coverage[col_name] = float(np.sum(valid) / len(valid)) if len(valid) > 0 else 0.0

        return coverage

    def coverage_report(self) -> str:
        """Print coverage report for all episodes."""
        report_lines = ["Coverage Report:", "================"]

        for ep_idx in range(self._df.num_episodes()):
            coverage = self.coverage_by_column(ep_idx)
            report_lines.append(f"\nEpisode {ep_idx}:")
            for col, cov in coverage.items():
                pct = cov * 100
                bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                report_lines.append(f"  {col:20} {bar} {pct:5.1f}%")

        return "\n".join(report_lines)

    def get_valid_indices(
        self, episode_index: int = 0, columns: list[str] | None = None
    ) -> np.ndarray:
        """Get indices where all specified columns are valid.

        Args:
            episode_index: Episode to check
            columns: Columns to check (default: all)

        Returns:
            Boolean array [N] indicating valid timesteps
        """
        ep_slice = self._df.slice(episode_index=episode_index)
        ep_table = ep_slice.to_pyarrow()

        if columns is None:
            columns = ep_table.column_names

        valid = np.ones(ep_table.num_rows, dtype=bool)

        for col_name in columns:
            if col_name not in ep_table.column_names:
                continue

            col = ep_table[col_name].combine_chunks().to_numpy()
            if col.dtype == object:
                col_valid = np.array([v is not None for v in col])
            elif np.issubdtype(col.dtype, np.floating):
                col_valid = ~np.isnan(col)
            else:
                col_valid = np.ones(len(col), dtype=bool)

            valid = valid & col_valid

        return valid


def interpolate_missing(
    dataframe: RoboticsDataFrame,
    method: Literal["forward_fill", "backward_fill", "linear", "nearest"] = "forward_fill",
    columns: list[str] | None = None,
    inplace: bool = False,
) -> RoboticsDataFrame:
    """Interpolate missing values in numeric columns.

    Args:
        dataframe: RoboticsDataFrame with missing data
        method: Interpolation method:
            - "forward_fill": last valid value
            - "backward_fill": next valid value
            - "linear": linear interpolation between valid points
            - "nearest": nearest valid value
        columns: Columns to interpolate (default: all numeric)
        inplace: Modify dataframe in-place (default: return copy)

    Returns:
        DataFrame with interpolated values
    """
    # This is a placeholder; full implementation would iterate episodes
    # and apply interpolation per column
    if inplace:
        return dataframe
    else:
        return dataframe  # Return copy in practice


def fill_with_zeros(
    dataframe: RoboticsDataFrame,
    columns: list[str] | None = None,
) -> RoboticsDataFrame:
    """Fill missing values with zeros (e.g., for zero-velocity assumption).

    Args:
        dataframe: RoboticsDataFrame
        columns: Columns to fill (default: action columns)

    Returns:
        DataFrame with NaN → 0
    """
    # Placeholder; full implementation would handle per-episode
    if columns is None:
        columns = [c for c in dataframe.column_names if "action" in c]
    return dataframe


class SensorHealthMonitor:
    """Track sensor health and failure patterns over time."""

    def __init__(self, dataframe: RoboticsDataFrame):
        """Initialize monitor.

        Args:
            dataframe: RoboticsDataFrame to monitor
        """
        self._df = dataframe

    def failure_rate_by_column(self) -> dict[str, float]:
        """Compute failure rate (% missing) for each column across all episodes."""
        all_coverage = {}

        for ep_idx in range(self._df.num_episodes()):
            ep_coverage = MaskedDataFrame(self._df).coverage_by_column(ep_idx)
            for col, cov in ep_coverage.items():
                if col not in all_coverage:
                    all_coverage[col] = []
                all_coverage[col].append(cov)

        # Summarize: mean coverage → failure rate
        return {
            col: 1.0 - np.mean(covs) for col, covs in all_coverage.items()
        }

    def failure_episodes(self, threshold: float = 0.1) -> list[int]:
        """Get episodes with > threshold missing data.

        Args:
            threshold: Failure rate threshold (e.g., 0.1 = 10% missing)

        Returns:
            Episode indices to exclude from training
        """
        failed = []
        for ep_idx in range(self._df.num_episodes()):
            masked = MaskedDataFrame(self._df)
            coverage = masked.coverage_by_column(ep_idx)
            failure_rate = 1.0 - np.mean(list(coverage.values()))
            if failure_rate > threshold:
                failed.append(ep_idx)
        return failed
