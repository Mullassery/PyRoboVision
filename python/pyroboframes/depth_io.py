"""Utilities for loading and converting depth data formats.

Includes helpers for:
- Loading NumPy arrays as point clouds
- Converting between depth map and point cloud formats
- Filtering and downsampling point clouds
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import _core


def load_point_cloud_from_numpy(
    points: np.ndarray,
    name: str = "depth_data",
) -> _core.PointCloudPy:
    """Create a PointCloud from a NumPy array.

    Args:
        points: Point positions as NumPy array:
            - Shape [N, 3]: N points with (x, y, z) coordinates
            - Shape [H, W, 3]: Depth map as H×W grid (flattened to N=H*W points)
            - dtype: float32 or float64 (converted to float32 internally)
        name: Optional name for the point cloud (for reference)

    Returns:
        PointCloud object containing the points

    Example:
        ```python
        # From a 3D point array
        points = np.random.randn(1000, 3).astype(np.float32)
        cloud = load_point_cloud_from_numpy(points)

        # From a depth map (480×640 pixels)
        depth_map = np.random.rand(480, 640).astype(np.float32)  # in meters
        cloud = load_point_cloud_from_depth_map(depth_map, fx=500, fy=500, cx=320, cy=240)
        ```
    """
    points = np.asarray(points, dtype=np.float32)

    if points.ndim == 3 and points.shape[2] == 3:
        # Depth map format [H, W, 3] -> flatten to [H*W, 3]
        h, w, _ = points.shape
        points_flat = points.reshape(h * w, 3)
    elif points.ndim == 2 and points.shape[1] == 3:
        # Already in correct format [N, 3]
        points_flat = points
    else:
        raise ValueError(
            f"Expected shape [N, 3] or [H, W, 3], got {points.shape}"
        )

    # Create PointCloud by saving to temp file and loading
    # (since Rust PointCloud is loaded from files, not constructed from arrays)
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        for point in points_flat:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        temp_path = f.name

    try:
        cloud = _core.PointCloudPy.load(temp_path)
    finally:
        Path(temp_path).unlink()

    return cloud


def load_point_cloud_from_depth_map(
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    name: str = "depth_map",
) -> _core.PointCloudPy:
    """Convert a depth map to a point cloud using camera intrinsics.

    Unprojected each pixel using the intrinsic matrix K:
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    Args:
        depth_map: Depth values in meters, shape [H, W]
        fx, fy: Focal lengths (pixels)
        cx, cy: Principal point (pixels)
        name: Optional name for reference

    Returns:
        PointCloud object

    Example:
        ```python
        depth = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        fx, fy = 500, 500
        cx, cy = 320, 240
        cloud = load_point_cloud_from_depth_map(depth, fx, fy, cx, cy)
        ```
    """
    depth_map = np.asarray(depth_map, dtype=np.float32)
    if depth_map.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth_map.shape}")

    h, w = depth_map.shape

    # Create point grid
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Unproject using intrinsics
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack into [H, W, 3]
    points = np.stack([x, y, z], axis=-1)

    return load_point_cloud_from_numpy(points, name=name)


def downsample_point_cloud(
    cloud: _core.PointCloudPy,
    factor: int = 2,
) -> _core.PointCloudPy:
    """Downsample a point cloud by taking every Nth point.

    Args:
        cloud: Input point cloud
        factor: Keep every Nth point (e.g., factor=2 keeps every 2nd point)

    Returns:
        Downsampled point cloud

    Example:
        ```python
        cloud = prf.PointCloud.load("large.pcd")
        cloud_sparse = downsample_point_cloud(cloud, factor=4)
        ```
    """
    if factor < 1:
        raise ValueError(f"Downsample factor must be >= 1, got {factor}")

    points = cloud.points()  # [N, 3]
    points_down = points[::factor]

    return load_point_cloud_from_numpy(points_down)


def filter_point_cloud(
    cloud: _core.PointCloudPy,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> _core.PointCloudPy:
    """Filter point cloud by depth range.

    Useful for removing sensor artifacts (very close points, far noise).

    Args:
        cloud: Input point cloud
        min_depth: Minimum distance from origin (meters)
        max_depth: Maximum distance from origin (meters)

    Returns:
        Filtered point cloud

    Example:
        ```python
        cloud = prf.PointCloud.load("scan.pcd")
        cloud_clean = filter_point_cloud(cloud, min_depth=0.1, max_depth=5.0)
        ```
    """
    points = cloud.points()  # [N, 3]
    depth = np.linalg.norm(points, axis=1)

    mask = (depth >= min_depth) & (depth <= max_depth)
    points_filtered = points[mask]

    return load_point_cloud_from_numpy(points_filtered)


def align_point_clouds_icp(
    source: _core.PointCloudPy,
    target: _core.PointCloudPy,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Align two point clouds using Iterative Closest Point (ICP).

    Returns the rotation matrix R and translation vector t such that:
    target ≈ R @ source + t

    Note: This is a basic implementation. For production use, consider:
    - Open3D: open3d.pipelines.registration.registration_icp()
    - PCL: pcl.registration.icp()

    Args:
        source: Point cloud to align (will be transformed)
        target: Reference point cloud
        max_iterations: Maximum ICP iterations
        tolerance: Convergence tolerance

    Returns:
        (R, t): Rotation matrix [3, 3] and translation vector [3]

    Raises:
        ImportError: If scipy is not available (required for ICP)
    """
    try:
        from scipy.spatial.transform import Rotation
        from scipy.spatial import KDTree
    except ImportError:
        raise ImportError(
            "ICP requires scipy. Install with: pip install scipy"
        )

    source_pts = source.points()  # [N, 3]
    target_pts = target.points()  # [M, 3]

    # Initialize with identity
    R = np.eye(3)
    t = np.zeros(3)

    for iteration in range(max_iterations):
        # Transform source points
        source_transformed = (R @ source_pts.T).T + t

        # Find nearest neighbors in target
        tree = KDTree(target_pts)
        distances, indices = tree.query(source_transformed)

        # Compute centroids
        source_center = source_pts.mean(axis=0)
        target_center = target_pts[indices].mean(axis=0)

        # Center the point sets
        source_centered = source_pts - source_center
        target_centered = target_pts[indices] - target_center

        # Compute cross-covariance matrix
        H = source_centered.T @ target_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)
        R_new = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_new) < 0:
            Vt[-1, :] *= -1
            R_new = Vt.T @ U.T

        t_new = target_center - R_new @ source_center

        # Check convergence
        delta_R = np.linalg.norm(R_new - R)
        delta_t = np.linalg.norm(t_new - t)

        if delta_R < tolerance and delta_t < tolerance:
            break

        R = R_new
        t = t_new

    return R, t
