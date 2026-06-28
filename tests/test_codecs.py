"""Tests for video codec selection and metadata handling."""

import json
import tempfile

import numpy as np
import pytest

import pyroboframes as prf


def test_write_with_default_codec():
    """Test that default codec is H.264 when not specified."""
    with tempfile.TemporaryDirectory() as tmp:
        features = {
            "observation.state": np.zeros((100, 7), dtype=np.float32),
            "action": np.zeros((100, 7), dtype=np.float32),
        }
        prf.write_lerobot_dataset(tmp, features, [50, 50], fps=30.0)

        # Verify codec in metadata
        with open(f"{tmp}/meta/info.json") as f:
            info = json.load(f)
        assert info["video_codec"] == "h264"
        assert "video_profile" not in info


def test_write_with_hevc_codec():
    """Test that HEVC codec is correctly stored in metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        features = {
            "observation.state": np.zeros((100, 7), dtype=np.float32),
            "action": np.zeros((100, 7), dtype=np.float32),
        }
        prf.write_lerobot_dataset(
            tmp, features, [50, 50], fps=30.0, video_codec="hevc", video_profile="main"
        )

        with open(f"{tmp}/meta/info.json") as f:
            info = json.load(f)
        assert info["video_codec"] == "hevc"
        assert info["video_profile"] == "main"


def test_write_with_av1_codec():
    """Test that AV1 codec is correctly stored in metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        features = {
            "observation.state": np.zeros((100, 7), dtype=np.float32),
            "action": np.zeros((100, 7), dtype=np.float32),
        }
        prf.write_lerobot_dataset(tmp, features, [50, 50], fps=30.0, video_codec="av1")

        with open(f"{tmp}/meta/info.json") as f:
            info = json.load(f)
        assert info["video_codec"] == "av1"
        assert "video_profile" not in info


def test_write_rejects_invalid_codec():
    """Test that invalid codec values are rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        features = {
            "observation.state": np.zeros((100, 7), dtype=np.float32),
            "action": np.zeros((100, 7), dtype=np.float32),
        }
        with pytest.raises(ValueError, match="video_codec must be"):
            prf.write_lerobot_dataset(
                tmp, features, [50, 50], fps=30.0, video_codec="invalid"
            )


def test_load_dataset_with_codec_metadata():
    """Test that codec metadata is preserved in the dataset JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        # Write dataset with HEVC
        features_in = {
            "observation.state": np.random.randn(100, 7).astype(np.float32),
            "action": np.random.randn(100, 7).astype(np.float32),
        }
        prf.write_lerobot_dataset(
            tmp, features_in, [50, 50], fps=30.0, video_codec="hevc", video_profile="main"
        )

        # Load dataset (verifies no errors)
        ds = prf.RoboFrameDataset.from_path(tmp)
        assert ds.num_frames == 100

        # Verify metadata was preserved in JSON
        with open(f"{tmp}/meta/info.json") as f:
            info = json.load(f)
        assert info["video_codec"] == "hevc"
        assert info["video_profile"] == "main"


def test_codec_backwards_compatibility():
    """Test that datasets without codec metadata default to H.264."""
    with tempfile.TemporaryDirectory() as tmp:
        # Write dataset without specifying codec (uses default)
        features = {
            "observation.state": np.zeros((100, 7), dtype=np.float32),
            "action": np.zeros((100, 7), dtype=np.float32),
        }
        prf.write_lerobot_dataset(tmp, features, [50, 50], fps=30.0)

        # Load and verify codec defaults to h264
        ds = prf.RoboFrameDataset.from_path(tmp)
        assert ds.num_frames == 100  # Dataset loads successfully

        # Verify codec field in metadata
        with open(f"{tmp}/meta/info.json") as f:
            info = json.load(f)
        assert info["video_codec"] == "h264"
