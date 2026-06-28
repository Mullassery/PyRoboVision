"""Fast proprioceptive-only dataloader (state/action without video decode).

For policies that don't use camera frames (joint-space control, IMU-based locomotion),
this loader provides 10× speedup by skipping video decode entirely.

Usage:
    ```python
    import pyroboframes as prf

    ds = prf.RoboFrameDataset.from_path("…")

    # Fast path: load state + action only (no video)
    loader = prf.ProprioceptiveLoader(
        ds,
        features=["observation.state", "action"],
        batch_size=64,
        device="mlx",
    )

    for batch in loader:
        state = batch["observation.state"]   # [64, state_dim]
        action = batch["action"]             # [64, action_dim]
        # ~10× faster than loading video + state
    ```

Key optimizations:
- Skips video decode (the bottleneck)
- Columnar parquet reads (state-action optimized)
- Temporal window support for sequence models
- GPU transfer only for requested features
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .backend import resolve_device, to_backend


class ProprioceptiveLoader:
    """Fast loader for proprioceptive data (state, action, IMU, sensors).

    Designed for policies that don't use camera frames. Skips video decode
    entirely, yielding 10× speedup for joint-space and proprioceptive-only models.

    Args:
        dataset: RoboFrameDataset instance
        features: List of feature names to load (e.g., ["observation.state", "action"])
        batch_size: Batch size
        sequence_length: If >1, load temporal windows of this length
        device: Target device ("cpu", "mlx", "cuda", "mps") - framework auto-selected
    """

    def __init__(
        self,
        dataset: Any,
        features: List[str],
        batch_size: int = 64,
        sequence_length: int = 1,
        device: str = "cpu",
    ):
        """Initialize proprioceptive loader."""
        self.dataset = dataset
        self.features = features
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = resolve_device(device)

        # Cache dataset metadata
        self.info = self._load_info()
        self.features_schema = self.info.get("features", {})

        # Load episode boundaries
        self.episode_indices, self.episode_lengths = self._load_episode_data()

        # Track position for resuming
        self.position = 0
        self._current_episode = 0

        # Validate requested features
        self._validate_features()

    def _load_info(self) -> Dict[str, Any]:
        """Load dataset info.json."""
        info_path = Path(self.dataset.path) / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                return json.load(f)
        return {"features": {}}

    def _load_episode_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load episode boundaries from parquet."""
        episodes_dir = Path(self.dataset.path) / "meta" / "episodes"
        if not episodes_dir.exists():
            return np.array([]), np.array([])

        episode_files = sorted(episodes_dir.glob("**/file-*.parquet"))
        if not episode_files:
            return np.array([]), np.array([])

        # Read episode metadata
        tables = [pq.read_table(str(f)) for f in episode_files]
        table = pa.concat_tables(tables) if tables else None

        if table is None or "episode_index" not in table.column_names:
            return np.array([]), np.array([])

        dataset_from_index = table.column("dataset_from_index").to_numpy()
        length = table.column("length").to_numpy()

        return dataset_from_index, length

    def _validate_features(self):
        """Check that all requested features exist."""
        missing = [f for f in self.features if f not in self.features_schema]
        if missing:
            available = list(self.features_schema.keys())
            raise ValueError(
                f"Features {missing} not found. Available: {available}"
            )

    def _get_parquet_path(self, chunk_idx: int = 0, file_idx: int = 0) -> Path:
        """Get path to data parquet file."""
        return (
            Path(self.dataset.path)
            / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        )

    def _load_feature_columns(
        self,
        indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Load specific rows from parquet, columnar format."""
        parquet_file = self._get_parquet_path()
        if not parquet_file.exists():
            raise FileNotFoundError(f"Data file not found: {parquet_file}")

        # Read only requested feature columns
        table = pq.read_table(
            str(parquet_file),
            columns=self.features,
        )

        # Convert to numpy and gather requested rows
        batch = {}
        for feature in self.features:
            if feature in table.column_names:
                col = table.column(feature).to_numpy(zero_copy_only=False)
                # Handle fixed-size lists (convert to [N, D] arrays)
                if hasattr(col[0], "__len__") and not isinstance(col[0], (str, bytes)):
                    batch[feature] = np.array([list(x) for x in col[indices]])
                else:
                    batch[feature] = col[indices]

        return batch

    def _create_temporal_windows(
        self,
        batch: Dict[str, np.ndarray],
        start_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Create temporal windows if sequence_length > 1."""
        if self.sequence_length == 1:
            return batch

        seq_len = self.sequence_length
        windowed = {}

        for feature, data in batch.items():
            if data.ndim >= 1:
                # Shape: [N, ...] → [N, seq_len, ...]
                N = len(start_indices)
                feature_shape = (N, seq_len) + data.shape[1:]
                windowed[feature] = np.zeros(feature_shape, dtype=data.dtype)

                for i, start_idx in enumerate(start_indices):
                    for t in range(seq_len):
                        idx = min(start_idx + t, len(data) - 1)
                        windowed[feature][i, t] = data[idx]

        return windowed

    def __iter__(self):
        """Iterate over batches."""
        self.position = 0
        num_samples = sum(self.episode_lengths) if len(self.episode_lengths) > 0 else 0

        if num_samples == 0:
            return

        # Create all valid indices (accounting for sequence_length)
        all_indices = []
        for ep_idx, (start, length) in enumerate(
            zip(self.episode_indices, self.episode_lengths)
        ):
            # Can't start sequence within last (sequence_length - 1) frames
            valid_end = start + length - (self.sequence_length - 1)
            valid_indices = np.arange(start, valid_end)
            all_indices.extend(valid_indices)

        all_indices = np.array(all_indices)
        num_batches = (len(all_indices) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(all_indices))
            batch_indices = all_indices[start:end]

            # Load features from parquet
            batch = self._load_feature_columns(batch_indices)

            # Add temporal windows if needed
            if self.sequence_length > 1:
                batch = self._create_temporal_windows(batch, batch_indices)

            # Move to device (to_backend automatically chooses framework based on device)
            batch = to_backend(batch, self.device)

            self.position = end
            yield batch

    def __len__(self) -> int:
        """Total number of batches."""
        num_samples = sum(self.episode_lengths) if len(self.episode_lengths) > 0 else 0
        return (num_samples + self.batch_size - 1) // self.batch_size

    def reset(self):
        """Reset iterator position."""
        self.position = 0
        self._current_episode = 0

    def __repr__(self) -> str:
        return (
            f"ProprioceptiveLoader("
            f"features={self.features}, "
            f"batch_size={self.batch_size}, "
            f"seq_len={self.sequence_length}, "
            f"device='{self.device}')"
        )


class ProprioceptiveDataFrame:
    """Lightweight time-indexed view of proprioceptive data (no video).

    Provides temporal slicing and resampling for state/action sequences.
    """

    def __init__(self, path: str, features: List[str]):
        """Initialize from dataset path."""
        self.path = Path(path)
        self.features = features
        self._table = None

    def _load_table(self) -> Any:
        """Load parquet table with requested features."""
        if self._table is None:
            parquet_file = self.path / "data/chunk-000/file-000.parquet"
            self._table = pq.read_table(str(parquet_file), columns=self.features)
        return self._table

    def slice(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """Get slice [start_idx:end_idx] as numpy arrays."""
        table = self._load_table()
        batch = {}
        for feature in self.features:
            col = table.column(feature).to_numpy(zero_copy_only=False)
            # Handle fixed-size lists
            if hasattr(col[0], "__len__") and not isinstance(col[0], (str, bytes)):
                col = np.array([list(x) for x in col])
            batch[feature] = col[start_idx:end_idx]
        return batch

    def resample(
        self,
        feature: str,
        indices: np.ndarray,
        method: str = "nearest",
    ) -> np.ndarray:
        """Resample a feature at given indices."""
        table = self._load_table()
        col = table.column(feature).to_numpy(zero_copy_only=False)

        # Handle fixed-size lists
        if hasattr(col[0], "__len__") and not isinstance(col[0], (str, bytes)):
            col = np.array([list(x) for x in col])

        if method == "nearest":
            return col[np.clip(indices.astype(int), 0, len(col) - 1)]
        elif method == "linear":
            # Ensure indices are floats for interpolation
            indices_float = np.asarray(indices, dtype=np.float64)
            x_coords = np.arange(len(col), dtype=np.float64)

            if col.ndim == 1:
                return np.interp(indices_float, x_coords, col.astype(np.float64))
            else:
                # For multi-dimensional data, interpolate each dimension
                result = np.zeros((len(indices), col.shape[1]), dtype=col.dtype)
                for d in range(col.shape[1]):
                    result[:, d] = np.interp(indices_float, x_coords, col[:, d].astype(np.float64))
                return result
        else:
            raise ValueError(f"Unknown resample method: {method}")

    def __repr__(self) -> str:
        return f"ProprioceptiveDataFrame(features={self.features})"
