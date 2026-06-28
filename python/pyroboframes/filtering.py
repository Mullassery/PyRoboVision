"""Episode filtering and querying API.

Provides SQL-like filtering on episode metadata for curriculum learning,
task-conditional training, and data curation.

```python
from pyroboframes.filtering import EpisodeFilter

# Create filter
filt = EpisodeFilter(df)

# SQL-like queries
success_episodes = filt.where(success=True)
pick_episodes = filt.where(task="pick", difficulty=["easy", "medium"])
long_episodes = filt.where(length_min=100, length_max=500)

# Combine filters
high_quality = filt.where(success=True).where(quality_score_min=0.7)

# Get loader with filtered episodes
loader = ds.loader(episodes=high_quality.episode_indices)
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

    from .dataframe import RoboticsDataFrame


class EpisodeFilter:
    """Build and apply filters on episode metadata."""

    def __init__(self, dataframe: RoboticsDataFrame):
        """Initialize filter for a RoboticsDataFrame.

        Args:
            dataframe: RoboticsDataFrame to filter
        """
        self._dataframe = dataframe
        self._episode_indices = list(range(dataframe.num_episodes()))
        self._conditions: list[tuple[str, str, Any]] = []

    def where(self, **kwargs) -> EpisodeFilter:
        """Apply filter conditions (SQL WHERE clause semantics).

        Supports:
        - Equality: `where(task="pick")`
        - Range: `where(length_min=50, length_max=200)`
        - Set membership: `where(difficulty=["easy", "medium"])`
        - Threshold: `where(quality_score_min=0.7)`

        Args:
            **kwargs: Filter conditions

        Returns:
            Self (for chaining)
        """
        for key, value in kwargs.items():
            if key.endswith("_min"):
                # Range minimum
                col = key[:-4]
                self._conditions.append((col, ">=", value))
            elif key.endswith("_max"):
                # Range maximum
                col = key[:-4]
                self._conditions.append((col, "<=", value))
            elif isinstance(value, (list, tuple)):
                # Set membership
                self._conditions.append((key, "in", value))
            else:
                # Equality
                self._conditions.append((key, "==", value))

        self._apply_conditions()
        return self

    def _apply_conditions(self) -> None:
        """Apply all accumulated conditions to filter episodes."""
        remaining = []

        for ep_idx in self._episode_indices:
            # Read episode metadata
            ep_slice = self._dataframe.slice(episode_index=ep_idx)
            ep_table = ep_slice.to_pyarrow()

            # Check all conditions
            passes_all = True
            for col, op, value in self._conditions:
                if not self._check_condition(ep_table, col, op, value):
                    passes_all = False
                    break

            if passes_all:
                remaining.append(ep_idx)

        self._episode_indices = remaining

    def _check_condition(self, table: pa.Table, col: str, op: str, value: Any) -> bool:
        """Check if episode passes a single condition."""
        if col not in table.column_names:
            # Column not in episode metadata; assume pass
            return True

        # Get column value (first row of metadata, or length from data)
        if col == "length":
            # Special case: episode length is total rows
            col_value = table.num_rows
        else:
            col_array = table[col].combine_chunks().to_pylist()
            if not col_array:
                return True
            col_value = col_array[0]

        # Apply condition
        if op == "==":
            return col_value == value
        elif op == "in":
            return col_value in value
        elif op == ">=":
            return col_value >= value
        elif op == "<=":
            return col_value <= value
        elif op == ">":
            return col_value > value
        elif op == "<":
            return col_value < value
        elif op == "!=":
            return col_value != value
        else:
            return True

    @property
    def episode_indices(self) -> list[int]:
        """List of episode indices passing all filters."""
        return self._episode_indices

    def count(self) -> int:
        """Number of episodes passing filters."""
        return len(self._episode_indices)

    def reset(self) -> EpisodeFilter:
        """Reset all filters."""
        self._episode_indices = list(range(self._dataframe.num_episodes()))
        self._conditions = []
        return self

    def __repr__(self) -> str:
        return f"EpisodeFilter(episodes={self.count()}, conditions={len(self._conditions)})"


class EpisodeFilterBuilder:
    """Fluent builder for complex filter chains."""

    def __init__(self, dataframe: RoboticsDataFrame):
        """Initialize builder.

        Args:
            dataframe: RoboticsDataFrame to filter
        """
        self._filters = [EpisodeFilter(dataframe)]

    def success_only(self) -> EpisodeFilterBuilder:
        """Keep only successful episodes."""
        self._filters[-1].where(success=True)
        return self

    def by_task(self, *tasks: str) -> EpisodeFilterBuilder:
        """Filter to specific task(s).

        Args:
            *tasks: One or more task names

        Returns:
            Self (for chaining)
        """
        if len(tasks) == 1:
            self._filters[-1].where(task=tasks[0])
        else:
            self._filters[-1].where(task=list(tasks))
        return self

    def by_difficulty(self, *difficulties: str) -> EpisodeFilterBuilder:
        """Filter by difficulty level(s).

        Args:
            *difficulties: One or more difficulty levels (e.g., "easy", "medium", "hard")

        Returns:
            Self (for chaining)
        """
        if len(difficulties) == 1:
            self._filters[-1].where(difficulty=difficulties[0])
        else:
            self._filters[-1].where(difficulty=list(difficulties))
        return self

    def min_length(self, length: int) -> EpisodeFilterBuilder:
        """Keep episodes with at least this many frames."""
        self._filters[-1].where(length_min=length)
        return self

    def max_length(self, length: int) -> EpisodeFilterBuilder:
        """Keep episodes with at most this many frames."""
        self._filters[-1].where(length_max=length)
        return self

    def min_quality(self, score: float) -> EpisodeFilterBuilder:
        """Keep episodes with quality_score >= threshold."""
        self._filters[-1].where(quality_score_min=score)
        return self

    def build(self) -> list[int]:
        """Build and return filtered episode indices."""
        return self._filters[-1].episode_indices

    def count(self) -> int:
        """Count filtered episodes."""
        return self._filters[-1].count()

    def __repr__(self) -> str:
        filt = self._filters[-1]
        return f"EpisodeFilterBuilder({filt.count()} episodes)"
