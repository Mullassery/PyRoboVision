"""Tests for P0: ProprioceptiveLoader (proprioceptive-only dataloader).

Tests fast path for state/action loading without video decode.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pyroboframes.proprioceptive_loader import ProprioceptiveLoader, ProprioceptiveDataFrame


@pytest.fixture
def mock_dataset():
    """Create a mock LeRobot-like dataset structure."""
    tmpdir = tempfile.TemporaryDirectory()
    path = Path(tmpdir.name)

    # Create directory structure
    (path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # Create mock info.json
    info = {
        "codebase_version": "v3.0",
        "fps": 30.0,
        "total_episodes": 2,
        "total_frames": 200,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [5]},
            "action": {"dtype": "float32", "shape": [3]},
            "observation.images.top": {"dtype": "uint8", "shape": [480, 640, 3]},
        },
    }
    with open(path / "meta" / "info.json", "w") as f:
        json.dump(info, f)

    # Create mock data parquet (state + action only, no video)
    state_dim, action_dim = 5, 3
    state_data = np.random.randn(200, state_dim).astype(np.float32)
    action_data = np.random.randn(200, action_dim).astype(np.float32)

    # Create fixed-size list arrays
    state_col = pa.FixedSizeListArray.from_arrays(
        pa.array(state_data.reshape(-1), type=pa.float32()), state_dim
    )
    action_col = pa.FixedSizeListArray.from_arrays(
        pa.array(action_data.reshape(-1), type=pa.float32()), action_dim
    )

    table = pa.table({"observation.state": state_col, "action": action_col})
    pq.write_table(table, path / "data" / "chunk-000" / "file-000.parquet")

    # Create episode metadata
    episode_indices = [0, 100]  # Two episodes
    episode_lengths = [100, 100]
    episode_table = pa.table(
        {
            "episode_index": pa.array([0, 1], pa.int64()),
            "length": pa.array(episode_lengths, pa.int64()),
            "dataset_from_index": pa.array(episode_indices, pa.int64()),
            "dataset_to_index": pa.array([100, 200], pa.int64()),
            "data/chunk_index": pa.array([0, 0], pa.int64()),
            "data/file_index": pa.array([0, 0], pa.int64()),
        }
    )
    pq.write_table(
        episode_table, path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )

    # Create mock dataset object
    class MockDataset:
        def __init__(self, path):
            self.path = str(path)

    return MockDataset(path), tmpdir


class TestProprioceptiveLoader:
    """Test ProprioceptiveLoader class."""

    def test_init(self, mock_dataset):
        """Test loader initialization."""
        ds, _ = mock_dataset
        loader = ProprioceptiveLoader(
            ds,
            features=["observation.state", "action"],
            batch_size=32,
        )

        assert loader.batch_size == 32
        assert loader.features == ["observation.state", "action"]
        assert loader.device == "cpu"

    def test_load_info(self, mock_dataset):
        """Test loading dataset info."""
        ds, _ = mock_dataset
        loader = ProprioceptiveLoader(ds, features=["observation.state"])

        assert "observation.state" in loader.features_schema
        assert loader.features_schema["observation.state"]["shape"] == [5]

    def test_validate_features(self, mock_dataset):
        """Test feature validation."""
        ds, _ = mock_dataset

        # Valid features
        loader = ProprioceptiveLoader(
            ds, features=["observation.state", "action"]
        )
        assert len(loader.features) == 2

        # Invalid feature
        with pytest.raises(ValueError, match="not found"):
            ProprioceptiveLoader(ds, features=["nonexistent.feature"])

    def test_iteration(self, mock_dataset):
        """Test iterating over batches."""
        ds, _ = mock_dataset
        loader = ProprioceptiveLoader(
            ds,
            features=["observation.state", "action"],
            batch_size=32,
            device="cpu",
        )

        batches = list(loader)
        assert len(batches) > 0

        # Check first batch
        batch = batches[0]
        assert "observation.state" in batch
        assert "action" in batch
        assert isinstance(batch["observation.state"], np.ndarray)
        assert batch["observation.state"].shape[0] <= 32
        assert batch["observation.state"].shape[1] == 5
        assert batch["action"].shape[1] == 3

    def test_batch_size(self, mock_dataset):
        """Test batch sizing."""
        ds, _ = mock_dataset
        loader = ProprioceptiveLoader(
            ds,
            features=["observation.state", "action"],
            batch_size=25,
            device="cpu",
        )

        batches = list(loader)
        total_samples = sum(b["observation.state"].shape[0] for b in batches)
        assert total_samples == 200  # Total frames in mock dataset

    def test_sequence_length(self, mock_dataset):
        """Test temporal window creation."""
        ds, _ = mock_dataset
        loader = ProprioceptiveLoader(
            ds,
            features=["observation.state"],
            batch_size=32,
            sequence_length=4,
            device="cpu",
        )

        batches = list(loader)
        if len(batches) > 0:
            batch = batches[0]
            state = batch["observation.state"]
            # With sequence_length=4, should have extra dimension
            assert state.ndim >= 2

    def test_position_tracking(self, mock_dataset):
        """Test position tracking during iteration."""
        ds, _ = mock_dataset
        loader = ProprioceptiveLoader(
            ds,
            features=["observation.state"],
            batch_size=32,
        )

        initial_position = loader.position
        assert initial_position == 0

        # Iterate and check position updates
        for _ in loader:
            assert loader.position > 0

    def test_reset(self, mock_dataset):
        """Test reset functionality."""
        ds, _ = mock_dataset
        loader = ProprioceptiveLoader(
            ds,
            features=["observation.state"],
            batch_size=32,
        )

        # Iterate once
        _ = list(loader)
        assert loader.position > 0

        # Reset
        loader.reset()
        assert loader.position == 0

    def test_len(self, mock_dataset):
        """Test __len__."""
        ds, _ = mock_dataset
        loader = ProprioceptiveLoader(
            ds,
            features=["observation.state"],
            batch_size=32,
        )

        length = len(loader)
        assert length > 0
        assert length == (200 + 32 - 1) // 32  # Expected batch count

    def test_repr(self, mock_dataset):
        """Test string representation."""
        ds, _ = mock_dataset
        loader = ProprioceptiveLoader(
            ds,
            features=["observation.state", "action"],
            batch_size=64,
        )

        repr_str = repr(loader)
        assert "ProprioceptiveLoader" in repr_str
        assert "observation.state" in repr_str
        assert "64" in repr_str


class TestProprioceptiveDataFrame:
    """Test ProprioceptiveDataFrame class."""

    def test_init(self, mock_dataset):
        """Test initialization."""
        ds, _ = mock_dataset
        df = ProprioceptiveDataFrame(
            ds.path,
            features=["observation.state", "action"],
        )

        assert df.features == ["observation.state", "action"]

    def test_slice(self, mock_dataset):
        """Test slicing."""
        ds, _ = mock_dataset
        df = ProprioceptiveDataFrame(ds.path, features=["observation.state"])

        batch = df.slice(0, 32)
        assert "observation.state" in batch
        assert batch["observation.state"].shape[0] == 32
        assert batch["observation.state"].shape[1] == 5

    def test_resample_nearest(self, mock_dataset):
        """Test resampling with nearest method."""
        ds, _ = mock_dataset
        df = ProprioceptiveDataFrame(ds.path, features=["observation.state"])

        indices = np.array([0, 10, 50, 99])
        resampled = df.resample("observation.state", indices, method="nearest")

        assert resampled.shape == (4, 5)

    def test_resample_linear(self, mock_dataset):
        """Test linear interpolation."""
        ds, _ = mock_dataset
        df = ProprioceptiveDataFrame(ds.path, features=["observation.state"])

        indices = np.array([0.5, 10.5, 50.5])
        resampled = df.resample("observation.state", indices, method="linear")

        assert resampled.shape[0] == 3

    def test_repr(self, mock_dataset):
        """Test string representation."""
        ds, _ = mock_dataset
        df = ProprioceptiveDataFrame(ds.path, features=["observation.state"])

        repr_str = repr(df)
        assert "ProprioceptiveDataFrame" in repr_str


class TestPerformance:
    """Basic performance tests."""

    def test_iteration_completes(self, mock_dataset):
        """Test that iteration completes without errors."""
        ds, _ = mock_dataset
        loader = ProprioceptiveLoader(
            ds,
            features=["observation.state", "action"],
            batch_size=32,
        )

        # Should complete without errors
        batch_count = 0
        for _ in loader:
            batch_count += 1

        assert batch_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
