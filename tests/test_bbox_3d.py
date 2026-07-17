import pytest
import numpy as np
from pyrobovision.perception.bbox_3d import BBox3D, Box3DConverter


class TestBBox3D:
    def test_initialization(self):
        center = np.array([1, 2, 3])
        dimensions = np.array([1, 1, 1])
        rotation = np.eye(3)

        bbox = BBox3D(
            center=center,
            dimensions=dimensions,
            rotation=rotation,
            confidence=0.9,
            class_id=0,
        )

        assert np.allclose(bbox.center, center)
        assert np.allclose(bbox.dimensions, dimensions)

    def test_volume(self):
        bbox = BBox3D(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 3, 4]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        volume = bbox.volume()
        assert volume == 24.0

    def test_to_corners(self):
        bbox = BBox3D(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 2, 2]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        corners = bbox.to_corners()
        assert corners.shape == (8, 3)

        expected_corner = np.array([-1, -1, -1])
        assert np.allclose(corners[0], expected_corner)

    def test_get_projection_2d(self):
        bbox = BBox3D(
            center=np.array([0, 0, 5]),
            dimensions=np.array([1, 1, 1]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        intrinsics = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

        projection = bbox.get_projection_2d(intrinsics)
        assert projection.shape == (8, 2)

    def test_intersection_over_union_identical(self):
        bbox = BBox3D(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 2, 2]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        iou = bbox.intersection_over_union(bbox)
        assert iou > 0.99

    def test_intersection_over_union_no_overlap(self):
        bbox1 = BBox3D(
            center=np.array([0, 0, 0]),
            dimensions=np.array([1, 1, 1]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        bbox2 = BBox3D(
            center=np.array([10, 10, 10]),
            dimensions=np.array([1, 1, 1]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        iou = bbox1.intersection_over_union(bbox2)
        assert iou == 0.0


class TestBox3DConverter:
    def test_initialization(self):
        converter = Box3DConverter()
        assert converter.intrinsics is not None

    def test_from_2d_bbox_and_depth(self):
        converter = Box3DConverter()

        bbox_2d = np.array([100, 100, 200, 200])
        depth_map = np.ones((480, 640)) * 5.0

        bbox_3d = converter.from_2d_bbox_and_depth(
            bbox_2d,
            depth_map,
            fx=500,
            fy=500,
            cx=320,
            cy=240,
            confidence=0.9,
        )

        assert bbox_3d.center is not None
        assert bbox_3d.dimensions is not None
        assert bbox_3d.confidence == 0.9

    def test_from_point_cloud_empty(self):
        converter = Box3DConverter()

        points = np.array([])

        bbox_3d = converter.from_point_cloud(points)

        assert bbox_3d.confidence == 0.0

    def test_from_point_cloud_valid(self):
        converter = Box3DConverter()

        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        bbox_3d = converter.from_point_cloud(points)

        assert bbox_3d.center is not None
        assert bbox_3d.dimensions is not None

    def test_transform_3d_bbox(self):
        converter = Box3DConverter()

        bbox = BBox3D(
            center=np.array([0, 0, 0]),
            dimensions=np.array([1, 1, 1]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        transform = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ])

        transformed = converter.transform_3d_bbox(bbox, transform)

        expected_center = np.array([1, 2, 3])
        assert np.allclose(transformed.center, expected_center)

    def test_compute_iou_3d(self):
        converter = Box3DConverter()

        bbox1 = BBox3D(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 2, 2]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        bbox2 = BBox3D(
            center=np.array([0, 0, 0]),
            dimensions=np.array([2, 2, 2]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        iou = converter.compute_iou_3d(bbox1, bbox2)
        assert iou > 0.99
