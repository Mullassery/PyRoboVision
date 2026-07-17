import pytest
import numpy as np
from pyrobovision.perception.depth import DepthEstimator, DepthMap


class TestDepthMap:
    def test_initialization(self):
        depth_data = np.random.rand(480, 640) * 10
        depth_map = DepthMap(
            data=depth_data,
            scale=0.001,
            shift=0.0,
            min_depth=0.1,
            max_depth=100.0,
        )

        assert depth_map.data.shape == (480, 640)
        assert depth_map.scale == 0.001

    def test_get_depth_valid(self):
        depth_data = np.ones((480, 640)) * 5.0
        depth_map = DepthMap(
            data=depth_data,
            scale=0.001,
            shift=0.0,
            min_depth=0.1,
            max_depth=100.0,
        )

        depth = depth_map.get_depth(320, 240)
        assert depth == 5.0

    def test_get_depth_invalid(self):
        depth_data = np.ones((480, 640)) * 5.0
        depth_map = DepthMap(
            data=depth_data,
            scale=0.001,
            shift=0.0,
            min_depth=0.1,
            max_depth=100.0,
        )

        depth = depth_map.get_depth(1000, 1000)
        assert depth == 100.0

    def test_get_depth_batch(self):
        depth_data = np.ones((480, 640)) * 5.0
        depth_map = DepthMap(
            data=depth_data,
            scale=0.001,
            shift=0.0,
            min_depth=0.1,
            max_depth=100.0,
        )

        points = np.array([[320, 240], [100, 100]])
        depths = depth_map.get_depth_batch(points)

        assert len(depths) == 2
        assert all(d == 5.0 for d in depths)

    def test_upsample(self):
        depth_data = np.ones((10, 10)) * 5.0
        depth_map = DepthMap(
            data=depth_data,
            scale=0.001,
            shift=0.0,
            min_depth=0.1,
            max_depth=100.0,
        )

        upsampled = depth_map.upsample(factor=2)
        assert upsampled.data.shape == (20, 20)

    def test_downsample(self):
        depth_data = np.ones((100, 100)) * 5.0
        depth_map = DepthMap(
            data=depth_data,
            scale=0.001,
            shift=0.0,
            min_depth=0.1,
            max_depth=100.0,
        )

        downsampled = depth_map.downsample(factor=2)
        assert downsampled.data.shape == (50, 50)

    def test_to_3d(self):
        depth_data = np.ones((100, 100)) * 5.0
        depth_map = DepthMap(
            data=depth_data,
            scale=0.001,
            shift=0.0,
            min_depth=0.1,
            max_depth=100.0,
        )

        points_3d = depth_map.to_3d(fx=500, fy=500, cx=50, cy=50)

        assert points_3d.shape[1] == 3
        assert len(points_3d) > 0


class TestDepthEstimator:
    def test_initialization(self):
        estimator = DepthEstimator(
            model="midas",
            min_depth=0.1,
            max_depth=100.0,
        )

        assert estimator.model == "midas"
        assert estimator.min_depth == 0.1

    def test_set_calibration(self):
        estimator = DepthEstimator()
        estimator.set_calibration(fx=500, fy=500, cx=320, cy=240)

        assert estimator.calibration_matrix is not None
        assert estimator.calibration_matrix.shape == (3, 3)

    def test_estimate_depth_rgb(self):
        estimator = DepthEstimator()

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        depth_map = estimator.estimate_depth(image)

        assert depth_map.data.shape == (480, 640)
        assert depth_map.min_depth <= depth_map.data.min()
        assert depth_map.data.max() <= depth_map.max_depth

    def test_estimate_depth_grayscale(self):
        estimator = DepthEstimator()

        image = np.random.rand(480, 640).astype(np.float32)

        depth_map = estimator.estimate_depth(image)

        assert depth_map.data.shape == (480, 640)

    def test_estimate_uncertainty(self):
        estimator = DepthEstimator()

        image = np.random.rand(100, 100).astype(np.float32)
        depth_map = estimator.estimate_depth(image)

        uncertainty = estimator.estimate_uncertainty(depth_map)

        assert uncertainty.shape == depth_map.data.shape
        assert np.all(uncertainty >= 0)

    def test_fuse_with_lidar(self):
        estimator = DepthEstimator()
        estimator.set_calibration(fx=500, fy=500, cx=320, cy=240)

        depth_data = np.ones((480, 640)) * 5.0
        depth_map = DepthMap(
            data=depth_data,
            scale=0.001,
            shift=0.0,
            min_depth=0.1,
            max_depth=100.0,
        )

        lidar_points = np.array([
            [0, 0, 5.0],
            [1, 1, 6.0],
            [2, 2, 4.5],
        ])

        fused = estimator.fuse_with_lidar(depth_map, lidar_points, weight=0.5)

        assert fused.data.shape == depth_map.data.shape
