import pytest
import numpy as np
from pyrobovision.perception.depth import DepthEstimator
from pyrobovision.perception.bbox_3d import Box3DConverter, BBox3D
from pyrobovision.perception.occupancy import OccupancyGridBuilder
from pyrobovision.perception.lidar import LiDARProcessor, PointCloud


class TestV15Integration:
    def test_monocular_depth_to_3d_bbox(self):
        estimator = DepthEstimator()
        converter = Box3DConverter()

        rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        depth_map = estimator.estimate_depth(rgb_image)

        bbox_2d = np.array([100, 100, 200, 200])
        bbox_3d = converter.from_2d_bbox_and_depth(
            bbox_2d,
            depth_map.data,
            fx=500,
            fy=500,
            cx=320,
            cy=240,
        )

        assert bbox_3d is not None
        assert bbox_3d.center is not None
        assert bbox_3d.dimensions is not None

    def test_lidar_point_cloud_to_occupancy(self):
        processor = LiDARProcessor()
        builder = OccupancyGridBuilder()

        raw_points = np.random.rand(1000, 3) * 20 - 10

        cloud = processor.process_raw(raw_points)

        occ_grid = builder.from_point_cloud(cloud.points)

        assert occ_grid.grid.shape == builder.grid_shape

    def test_3d_bbox_to_occupancy(self):
        builder = OccupancyGridBuilder()

        bboxes = [
            BBox3D(
                center=np.array([0, 0, 1]),
                dimensions=np.array([1, 1, 1]),
                rotation=np.eye(3),
                confidence=0.9,
            ),
            BBox3D(
                center=np.array([5, 5, 1]),
                dimensions=np.array([1, 1, 1]),
                rotation=np.eye(3),
                confidence=0.85,
            ),
        ]

        occ_grid = builder.from_3d_bboxes(bboxes)

        occupied = occ_grid.get_occupied_cells(threshold=0.5)
        assert len(occupied) >= 2

    def test_depth_lidar_fusion(self):
        estimator = DepthEstimator()
        estimator.set_calibration(fx=500, fy=500, cx=320, cy=240)

        processor = LiDARProcessor()

        rgb_image = np.random.rand(480, 640, 3).astype(np.float32)
        depth_map = estimator.estimate_depth(rgb_image)

        lidar_points = np.random.rand(100, 3) * 10

        fused = estimator.fuse_with_lidar(depth_map, lidar_points, weight=0.5)

        assert fused.data.shape == depth_map.data.shape

    def test_full_3d_perception_pipeline(self):
        estimator = DepthEstimator()
        estimator.set_calibration(fx=500, fy=500, cx=320, cy=240)

        converter = Box3DConverter()
        processor = LiDARProcessor()
        builder = OccupancyGridBuilder()

        rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        raw_lidar = np.random.rand(500, 3) * 20 - 10

        depth_map = estimator.estimate_depth(rgb_image)

        lidar_cloud = processor.process_raw(raw_lidar)
        fused_depth = estimator.fuse_with_lidar(depth_map, lidar_cloud.points)

        bbox_2d = np.array([100, 100, 200, 200])
        bbox_3d = converter.from_2d_bbox_and_depth(
            bbox_2d,
            fused_depth.data,
            fx=500,
            fy=500,
            cx=320,
            cy=240,
        )

        occ_grid = builder.from_3d_bboxes([bbox_3d])

        assert bbox_3d is not None
        assert occ_grid.grid.shape == builder.grid_shape

    def test_lidar_ground_removal_and_clustering(self):
        processor = LiDARProcessor()

        points = np.array([
            [0, 0, 0.1],
            [0.5, 0, 0.1],
            [1, 0, 0.1],
            [5, 5, 1.0],
            [5.1, 5, 1.0],
            [5.2, 5, 1.0],
            [10, 10, 2.0],
            [10.1, 10, 2.0],
            [10.2, 10, 2.0],
        ])
        cloud = PointCloud(points=points)

        non_ground = processor.remove_ground(cloud, ground_threshold=0.5)

        clusters = processor.cluster_points(non_ground, eps=1.0, min_points=2)

        assert len(non_ground) < len(cloud)
        assert len(clusters) > 0

    def test_multiple_3d_bboxes_iou(self):
        converter = Box3DConverter()

        bbox1 = BBox3D(
            center=np.array([0, 0, 1]),
            dimensions=np.array([1, 1, 1]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        bbox2 = BBox3D(
            center=np.array([0.5, 0, 1]),
            dimensions=np.array([1, 1, 1]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        iou = converter.compute_iou_3d(bbox1, bbox2)

        assert 0 <= iou <= 1

    def test_depth_uncertainty_map(self):
        estimator = DepthEstimator()

        rgb_image = np.random.rand(100, 100, 3).astype(np.float32)

        depth_map = estimator.estimate_depth(rgb_image)

        uncertainty = estimator.estimate_uncertainty(depth_map)

        assert uncertainty.shape == depth_map.data.shape
        assert np.all(uncertainty >= 0)

    def test_occupancy_grid_morphology(self):
        builder = OccupancyGridBuilder(grid_size=(50, 50))

        points = np.array([
            [0, 0, 1.0],
            [0.05, 0, 1.0],
            [0.1, 0, 1.0],
        ])

        occ_grid = builder.from_point_cloud(points)

        connectivity_before = occ_grid.get_connectivity_map()

        occ_grid.dilate(kernel_size=3)

        occupied_after = occ_grid.get_occupied_cells(threshold=0.5)

        assert len(occupied_after) > 0
