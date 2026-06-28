"""Tests for depth camera and point cloud support."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import pyroboframes as prf


def test_load_xyz_point_cloud():
    """Test loading a point cloud from XYZ format."""
    xyz_content = "0.0 0.0 0.0\n1.0 1.0 1.0\n2.0 2.0 2.0\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(xyz_content)
        path = f.name

    try:
        cloud = prf.PointCloud.load(path)
        assert len(cloud) == 3
        points = cloud.points()
        assert points.shape == (3, 3)
        assert np.allclose(points[0], [0.0, 0.0, 0.0])
        assert np.allclose(points[1], [1.0, 1.0, 1.0])
    finally:
        Path(path).unlink()


def test_load_xyz_with_comments():
    """Test loading XYZ with comment lines."""
    xyz_content = "# Point cloud\n0.0 0.0 0.0\n# comment\n1.0 1.0 1.0\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(xyz_content)
        path = f.name

    try:
        cloud = prf.PointCloud.load(path)
        assert len(cloud) == 2
    finally:
        Path(path).unlink()


def test_load_ply_point_cloud():
    """Test loading a point cloud from PLY format."""
    ply_content = """ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
end_header
0.0 0.0 0.0
1.0 1.0 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ply", delete=False) as f:
        f.write(ply_content)
        path = f.name

    try:
        cloud = prf.PointCloud.load(path)
        assert len(cloud) == 2
        points = cloud.points()
        assert np.allclose(points[0], [0.0, 0.0, 0.0])
        assert np.allclose(points[1], [1.0, 1.0, 1.0])
    finally:
        Path(path).unlink()


def test_load_pcd_point_cloud():
    """Test loading a point cloud from PCD format."""
    pcd_content = """VERSION 0.7
FIELDS X Y Z
SIZE 4 4 4
TYPE f f f
COUNT 1 1 1
WIDTH 2
HEIGHT 1
POINTS 2
DATA ascii
0.0 0.0 0.0
1.0 1.0 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pcd", delete=False) as f:
        f.write(pcd_content)
        path = f.name

    try:
        cloud = prf.PointCloud.load(path)
        assert len(cloud) == 2
        points = cloud.points()
        assert np.allclose(points[0], [0.0, 0.0, 0.0])
    finally:
        Path(path).unlink()


def test_rejects_unsupported_format():
    """Test that unsupported formats are rejected."""
    with pytest.raises(Exception):  # Expected to raise
        prf.PointCloud.load("test.unknown")


def test_empty_xyz_file():
    """Test that empty point cloud files are rejected."""
    xyz_content = "# Just comments\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(xyz_content)
        path = f.name

    try:
        with pytest.raises(Exception):  # Should raise "no valid points found"
            prf.PointCloud.load(path)
    finally:
        Path(path).unlink()


def test_point_cloud_is_empty_property():
    """Test the is_empty property."""
    xyz_content = "0.0 0.0 0.0\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(xyz_content)
        path = f.name

    try:
        cloud = prf.PointCloud.load(path)
        assert not cloud.is_empty
        assert len(cloud) == 1
    finally:
        Path(path).unlink()


def test_point_cloud_repr():
    """Test the string representation of a point cloud."""
    xyz_content = "0.0 0.0 0.0\n1.0 1.0 1.0\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(xyz_content)
        path = f.name

    try:
        cloud = prf.PointCloud.load(path)
        repr_str = repr(cloud)
        assert "PointCloud" in repr_str
        assert "points=2" in repr_str
    finally:
        Path(path).unlink()
