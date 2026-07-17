import pytest
import numpy as np
from pyrobovision.tracking.association import ObjectAssociation


class TestObjectAssociation:
    def test_initialization(self):
        oa = ObjectAssociation(iou_threshold=0.3, max_distance=100.0)
        assert oa.iou_threshold == 0.3
        assert oa.max_distance == 100.0

    def test_compute_iou_identical_boxes(self):
        oa = ObjectAssociation()
        box = np.array([0, 0, 10, 10])

        iou = oa.compute_iou(box, box)
        assert iou == 1.0

    def test_compute_iou_no_overlap(self):
        oa = ObjectAssociation()
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 30, 30])

        iou = oa.compute_iou(box1, box2)
        assert iou == 0.0

    def test_compute_iou_partial_overlap(self):
        oa = ObjectAssociation()
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([5, 5, 15, 15])

        iou = oa.compute_iou(box1, box2)
        assert 0 < iou < 1

    def test_compute_centroid_distance_same_box(self):
        oa = ObjectAssociation()
        box = np.array([0, 0, 10, 10])

        dist = oa.compute_centroid_distance(box, box)
        assert dist == 0.0

    def test_compute_centroid_distance_different_boxes(self):
        oa = ObjectAssociation()
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 30, 30])

        dist = oa.compute_centroid_distance(box1, box2)
        expected = np.sqrt((25 - 5)**2 + (25 - 5)**2)
        assert np.isclose(dist, expected)

    def test_build_cost_matrix_empty(self):
        oa = ObjectAssociation()
        cost_matrix = oa.build_cost_matrix([], [])
        assert cost_matrix.shape == (0, 0)

    def test_build_cost_matrix_single_track_detection(self):
        oa = ObjectAssociation()
        track_boxes = [np.array([0, 0, 10, 10])]
        detection_boxes = [np.array([2, 2, 12, 12])]

        cost_matrix = oa.build_cost_matrix(track_boxes, detection_boxes)
        assert cost_matrix.shape == (1, 1)
        assert cost_matrix[0, 0] < oa.max_distance

    def test_associate_perfect_match(self):
        oa = ObjectAssociation()
        track_boxes = [np.array([0, 0, 10, 10]), np.array([50, 50, 60, 60])]
        detection_boxes = [np.array([1, 1, 11, 11]), np.array([51, 51, 61, 61])]

        matches, unmatched_tracks, unmatched_detections = oa.associate(track_boxes, detection_boxes)

        assert len(matches) == 2
        assert len(unmatched_tracks) == 0
        assert len(unmatched_detections) == 0

    def test_associate_no_match(self):
        oa = ObjectAssociation()
        track_boxes = [np.array([0, 0, 10, 10])]
        detection_boxes = [np.array([100, 100, 110, 110])]

        matches, unmatched_tracks, unmatched_detections = oa.associate(track_boxes, detection_boxes)

        assert len(matches) == 0
        assert len(unmatched_tracks) == 1
        assert len(unmatched_detections) == 1

    def test_associate_more_detections_than_tracks(self):
        oa = ObjectAssociation()
        track_boxes = [np.array([0, 0, 10, 10])]
        detection_boxes = [np.array([1, 1, 11, 11]), np.array([50, 50, 60, 60])]

        matches, unmatched_tracks, unmatched_detections = oa.associate(track_boxes, detection_boxes)

        assert len(matches) == 1
        assert len(unmatched_tracks) == 0
        assert len(unmatched_detections) == 1

    def test_associate_more_tracks_than_detections(self):
        oa = ObjectAssociation()
        track_boxes = [np.array([0, 0, 10, 10]), np.array([50, 50, 60, 60])]
        detection_boxes = [np.array([1, 1, 11, 11])]

        matches, unmatched_tracks, unmatched_detections = oa.associate(track_boxes, detection_boxes)

        assert len(matches) == 1
        assert len(unmatched_tracks) == 1
        assert len(unmatched_detections) == 0

    def test_associate_empty_tracks(self):
        oa = ObjectAssociation()
        track_boxes = []
        detection_boxes = [np.array([0, 0, 10, 10])]

        matches, unmatched_tracks, unmatched_detections = oa.associate(track_boxes, detection_boxes)

        assert len(matches) == 0
        assert len(unmatched_tracks) == 0
        assert len(unmatched_detections) == 1

    def test_associate_empty_detections(self):
        oa = ObjectAssociation()
        track_boxes = [np.array([0, 0, 10, 10])]
        detection_boxes = []

        matches, unmatched_tracks, unmatched_detections = oa.associate(track_boxes, detection_boxes)

        assert len(matches) == 0
        assert len(unmatched_tracks) == 1
        assert len(unmatched_detections) == 0
