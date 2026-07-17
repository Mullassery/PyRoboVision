import pytest
import numpy as np
from pyrobovision.perception.lidar import PointCloud, LiDARProcessor


class TestPointCloud:
    def test_initialization(self):
        points = np.random.rand(1000, 3)
        cloud = PointCloud(points=points)

        assert len(cloud) == 1000
        assert cloud.points.shape == (1000, 3)

    def test_with_intensities(self):
        points = np.random.rand(1000, 3)
        intensities = np.random.rand(1000)

        cloud = PointCloud(points=points, intensities=intensities)

        assert cloud.intensities is not None
        assert len(cloud.intensities) == 1000

    def test_filter_by_range_empty(self):
        points = np.array([[100, 100, 100]])
        cloud = PointCloud(points=points)

        filtered = cloud.filter_by_range(0.1, 10.0)

        assert len(filtered) == 0

    def test_filter_by_range_valid(self):
        points = np.array([[1, 0, 0], [0.5, 0, 0], [10, 0, 0]])
        cloud = PointCloud(points=points)

        filtered = cloud.filter_by_range(0.1, 2.0)

        assert len(filtered) == 2

    def test_filter_by_height(self):
        points = np.array([[0, 0, 0.5], [0, 0, 1.5], [0, 0, 3.0]])
        cloud = PointCloud(points=points)

        filtered = cloud.filter_by_height(0.0, 2.0)

        assert len(filtered) == 2

    def test_voxelize(self):
        points = np.array([
            [0, 0, 0],
            [0.05, 0.05, 0.05],
            [1, 1, 1],
        ])
        cloud = PointCloud(points=points)

        voxelized = cloud.voxelize(voxel_size=0.1)

        assert len(voxelized) <= len(cloud)

    def test_downsample(self):
        points = np.random.rand(1000, 3)
        cloud = PointCloud(points=points)

        downsampled = cloud.downsample(num_points=100)

        assert len(downsampled) == 100

    def test_get_statistics(self):
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        cloud = PointCloud(points=points)

        stats = cloud.get_statistics()

        assert stats["num_points"] == 3
        assert "center" in stats
        assert "range" in stats


class TestLiDARProcessor:
    def test_initialization(self):
        processor = LiDARProcessor(
            min_range=0.5,
            max_range=100.0,
            min_height=-1.0,
            max_height=5.0,
        )

        assert processor.min_range == 0.5
        assert processor.max_range == 100.0

    def test_process_raw(self):
        processor = LiDARProcessor()

        raw_points = np.array([
            [1, 0, 0],
            [0.1, 0, 0],
            [100, 0, 0],
            [0, 0, 10],
        ])

        cloud = processor.process_raw(raw_points)

        assert len(cloud) <= len(raw_points)

    def test_remove_ground(self):
        processor = LiDARProcessor()

        points = np.array([
            [0, 0, 0.1],
            [1, 1, 0.2],
            [2, 2, 1.0],
            [3, 3, 2.0],
        ])
        cloud = PointCloud(points=points)

        non_ground = processor.remove_ground(cloud, ground_threshold=0.5)

        assert len(non_ground) < len(cloud)

    def test_cluster_points_empty(self):
        processor = LiDARProcessor()

        cloud = PointCloud(points=np.array([]))

        clusters = processor.cluster_points(cloud)

        assert len(clusters) == 0

    def test_cluster_points_single_cluster(self):
        processor = LiDARProcessor()

        points = np.array([
            [0, 0, 0],
            [0.1, 0, 0],
            [0.2, 0, 0],
            [0.3, 0, 0],
            [0.4, 0, 0],
        ])
        cloud = PointCloud(points=points)

        clusters = processor.cluster_points(cloud, eps=0.5, min_points=3)

        assert len(clusters) >= 1

    def test_cluster_points_multiple_clusters(self):
        processor = LiDARProcessor()

        points = np.array([
            [0, 0, 0],
            [0.1, 0, 0],
            [0.2, 0, 0],
            [10, 10, 0],
            [10.1, 10, 0],
            [10.2, 10, 0],
        ])
        cloud = PointCloud(points=points)

        clusters = processor.cluster_points(cloud, eps=0.5, min_points=2)

        assert len(clusters) >= 1

    def test_compute_normals(self):
        processor = LiDARProcessor()

        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ])
        cloud = PointCloud(points=points)

        normals = processor.compute_normals(cloud, k=3)

        assert normals.shape == points.shape
        assert all(np.isfinite(n).all() for n in normals)

    def test_transform_points(self):
        processor = LiDARProcessor()

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        cloud = PointCloud(points=points)

        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]

        transformed = processor.transform_points(cloud, transform)

        assert transformed.points.shape == cloud.points.shape
        assert np.allclose(transformed.points[0], [1, 2, 3])
