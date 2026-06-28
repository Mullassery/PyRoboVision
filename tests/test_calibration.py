"""Tests for camera calibration (intrinsics, distortion, poses)."""

import numpy as np
import pytest

import pyroboframes as prf


def test_camera_intrinsics_creation():
    """Test creating camera intrinsics."""
    intr = prf.CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    assert intr.fx == 500.0
    assert intr.fy == 500.0
    assert intr.cx == 320.0
    assert intr.cy == 240.0
    assert intr.width == 640
    assert intr.height == 480


def test_camera_intrinsics_k_matrix():
    """Test K matrix extraction."""
    intr = prf.CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    k = intr.k_matrix()
    assert k.shape == (3, 3)
    assert k[0, 0] == 500.0  # fx
    assert k[1, 1] == 500.0  # fy
    assert k[0, 2] == 320.0  # cx
    assert k[1, 2] == 240.0  # cy
    assert k[2, 2] == 1.0


def test_camera_intrinsics_project():
    """Test projecting 3D points to image plane."""
    intr = prf.CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    # Point at (1, 1, 2) meters
    u, v = intr.project(1.0, 1.0, 2.0)
    # u = 500 * (1/2) + 320 = 570
    # v = 500 * (1/2) + 240 = 490
    assert np.isclose(u, 570.0, atol=0.1)
    assert np.isclose(v, 490.0, atol=0.1)


def test_camera_intrinsics_project_behind_camera():
    """Test that points behind camera return None."""
    intr = prf.CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    assert intr.project(1.0, 1.0, -1.0) is None
    assert intr.project(1.0, 1.0, 0.0) is None


def test_camera_intrinsics_unproject_direction():
    """Test unprojecting pixel to 3D ray direction."""
    intr = prf.CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    # Unproject the principal point (should give [0, 0, 1] direction)
    direction = intr.unproject_direction(320.0, 240.0)
    assert np.allclose(direction, [0.0, 0.0, 1.0], atol=0.01)


def test_camera_calibration_creation():
    """Test creating camera calibration."""
    calib = prf.CameraCalibration(
        name="observation.images.top",
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    assert calib.name == "observation.images.top"
    assert calib.intrinsics.fx == 500.0


def test_camera_calibration_project_world_point():
    """Test projecting world 3D points through full calibration."""
    calib = prf.CameraCalibration(
        name="observation.images.top",
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    # Point at (1, 1, 2) in world frame (same as camera frame since identity pose)
    u, v = calib.project_world_point(1.0, 1.0, 2.0)
    assert np.isclose(u, 570.0, atol=0.1)
    assert np.isclose(v, 490.0, atol=0.1)


def test_camera_calibration_unproject_to_world_ray():
    """Test unprojecting pixel to world ray."""
    calib = prf.CameraCalibration(
        name="observation.images.top",
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    # Unproject the principal point
    origin, direction = calib.unproject_to_world_ray(320.0, 240.0)
    # With identity pose, origin should be at [0, 0, 0] (camera position)
    assert np.allclose(origin, [0.0, 0.0, 0.0], atol=0.01)
    # Direction should be [0, 0, 1] (looking along +Z)
    assert np.allclose(direction, [0.0, 0.0, 1.0], atol=0.01)


def test_camera_calibration_repr():
    """Test string representation."""
    calib = prf.CameraCalibration(
        name="camera1",
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    repr_str = repr(calib)
    assert "CameraCalibration" in repr_str
    assert "camera1" in repr_str
    assert "640x480" in repr_str


def test_camera_intrinsics_repr():
    """Test string representation of intrinsics."""
    intr = prf.CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    repr_str = repr(intr)
    assert "CameraIntrinsics" in repr_str
    assert "500.0" in repr_str
    assert "640x480" in repr_str


def test_multi_camera_setup():
    """Test setting up multiple cameras for a robot."""
    # Typical robot with 3 cameras
    top_camera = prf.CameraCalibration(
        name="observation.images.top",
        fx=480.0,
        fy=480.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    front_camera = prf.CameraCalibration(
        name="observation.images.front",
        fx=500.0,
        fy=500.0,
        cx=352.0,
        cy=288.0,
        width=704,
        height=576,
    )
    wrist_camera = prf.CameraCalibration(
        name="observation.images.wrist",
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )

    # Verify all cameras loaded correctly
    assert top_camera.name == "observation.images.top"
    assert front_camera.name == "observation.images.front"
    assert wrist_camera.name == "observation.images.wrist"

    # Verify different focal lengths
    assert top_camera.intrinsics.fx == 480.0
    assert front_camera.intrinsics.fx == 500.0
    assert wrist_camera.intrinsics.fx == 600.0


def test_projection_back_to_world():
    """Test round-trip: project world point to image, then unproject back."""
    calib = prf.CameraCalibration(
        name="camera",
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )

    # Original 3D point in world
    world_point = (2.0, 1.0, 5.0)

    # Project to image
    u, v = calib.project_world_point(*world_point)

    # Unproject to ray
    origin, direction = calib.unproject_to_world_ray(u, v)

    # The world point should lie on the ray at some depth t
    # Point = origin + t * direction
    # For identity pose: point = [0,0,0] + t * direction
    # So direction should be proportional to world_point
    expected_dir = np.array(world_point) / np.linalg.norm(world_point)
    actual_dir = np.array(direction)

    assert np.allclose(actual_dir, expected_dir, atol=0.01)
