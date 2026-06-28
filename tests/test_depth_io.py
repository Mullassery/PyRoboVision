"""Tests for depth I/O utilities (conversion, filtering, alignment)."""

import numpy as np
import pytest

import pyroboframes as prf
from pyroboframes.depth_io import (
    downsample_point_cloud,
    filter_point_cloud,
    load_point_cloud_from_depth_map,
    load_point_cloud_from_numpy,
)


def test_load_point_cloud_from_numpy_nx3():
    """Test loading point cloud from [N, 3] array."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32
    )
    cloud = load_point_cloud_from_numpy(points)
    assert len(cloud) == 3


def test_load_point_cloud_from_numpy_hwx3():
    """Test loading point cloud from [H, W, 3] depth map."""
    # Create a simple 2×2 depth map
    points = np.array(
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    cloud = load_point_cloud_from_numpy(points)
    assert len(cloud) == 4  # 2×2 = 4 points


def test_load_point_cloud_from_numpy_invalid_shape():
    """Test that invalid shapes are rejected."""
    points = np.array([[1, 2], [3, 4]], dtype=np.float32)  # [2, 2]
    with pytest.raises(ValueError, match="Expected shape"):
        load_point_cloud_from_numpy(points)


def test_load_point_cloud_from_depth_map():
    """Test converting depth map to point cloud."""
    # Simple 3×3 depth map with constant depth=1.0
    depth_map = np.ones((3, 3), dtype=np.float32)
    fx, fy = 500.0, 500.0
    cx, cy = 1.0, 1.0  # Smaller resolution, so principal point at (1, 1)

    cloud = load_point_cloud_from_depth_map(depth_map, fx, fy, cx, cy)
    assert len(cloud) == 9  # 3×3 pixels

    # Verify a few points
    points = cloud.points()
    # Point at pixel (1, 1) should be at (0, 0, 1) in 3D (center, at principal point)
    assert np.allclose(points[4], [0.0, 0.0, 1.0], atol=0.01)  # Middle point


def test_downsample_point_cloud():
    """Test downsampling a point cloud."""
    points = np.array(
        [[i, i, i] for i in range(10)], dtype=np.float32
    )
    cloud = load_point_cloud_from_numpy(points)
    assert len(cloud) == 10

    # Downsample by 2x
    cloud_down = downsample_point_cloud(cloud, factor=2)
    assert len(cloud_down) == 5  # Every other point


def test_downsample_point_cloud_invalid_factor():
    """Test that invalid downsample factors are rejected."""
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    cloud = load_point_cloud_from_numpy(points)

    with pytest.raises(ValueError, match="factor must be >= 1"):
        downsample_point_cloud(cloud, factor=0)


def test_filter_point_cloud():
    """Test filtering point cloud by depth range."""
    # Create points at various distances
    points = np.array(
        [
            [0.0, 0.0, 0.5],   # depth=0.5
            [0.0, 0.0, 1.0],   # depth=1.0
            [0.0, 0.0, 2.0],   # depth=2.0
            [0.0, 0.0, 10.0],  # depth=10.0
            [0.0, 0.0, 15.0],  # depth=15.0
        ],
        dtype=np.float32,
    )
    cloud = load_point_cloud_from_numpy(points)
    assert len(cloud) == 5

    # Filter: keep 1.0 to 10.0
    cloud_filt = filter_point_cloud(cloud, min_depth=1.0, max_depth=10.0)
    # Should have points at depths 1.0, 2.0, 10.0 (3 points)
    assert len(cloud_filt) == 3


def test_filter_point_cloud_empty():
    """Test filtering that results in empty cloud raises error."""
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    cloud = load_point_cloud_from_numpy(points)

    # Filter with impossible range should raise error (can't have empty cloud)
    with pytest.raises(ValueError, match="no valid points"):
        filter_point_cloud(cloud, min_depth=10.0, max_depth=20.0)


def test_round_trip_numpy_to_cloud():
    """Test converting numpy array to cloud and back."""
    original_points = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )
    cloud = load_point_cloud_from_numpy(original_points)
    recovered_points = cloud.points()

    assert np.allclose(original_points, recovered_points, atol=0.0001)


def test_depth_map_grid_structure():
    """Test that depth map preserves grid structure in flattening."""
    # 2×3 depth map with known pattern
    depth = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )
    # Create [H, W, 3] by stacking depth as Z
    points_3d = np.stack([
        np.zeros_like(depth),  # x
        np.zeros_like(depth),  # y
        depth,                  # z = depth
    ], axis=-1)

    cloud = load_point_cloud_from_numpy(points_3d)
    recovered = cloud.points()

    # Points should be flattened row-by-row
    expected_z = [1, 2, 3, 4, 5, 6]
    assert np.allclose(recovered[:, 2], expected_z, atol=0.0001)
