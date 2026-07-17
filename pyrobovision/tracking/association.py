import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment


class ObjectAssociation:
    def __init__(self, iou_threshold: float = 0.3, max_distance: float = 100.0):
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance

    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        intersection_x_min = max(x1_min, x2_min)
        intersection_y_min = max(y1_min, y2_min)
        intersection_x_max = min(x1_max, x2_max)
        intersection_y_max = min(y1_max, y2_max)

        if intersection_x_max < intersection_x_min or intersection_y_max < intersection_y_min:
            return 0.0

        intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def compute_centroid_distance(self, box1: np.ndarray, box2: np.ndarray) -> float:
        c1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
        c2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
        return float(np.linalg.norm(c1 - c2))

    def build_cost_matrix(self, track_boxes: List[np.ndarray], detection_boxes: List[np.ndarray]) -> np.ndarray:
        n_tracks = len(track_boxes)
        n_detections = len(detection_boxes)
        cost_matrix = np.full((n_tracks, n_detections), self.max_distance)

        for i, track_box in enumerate(track_boxes):
            for j, det_box in enumerate(detection_boxes):
                iou = self.compute_iou(track_box, det_box)
                if iou > self.iou_threshold:
                    distance = self.compute_centroid_distance(track_box, det_box)
                    cost_matrix[i, j] = distance * (1 - iou)

        return cost_matrix

    def associate(self, track_boxes: List[np.ndarray], detection_boxes: List[np.ndarray]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(track_boxes) == 0:
            return [], [], list(range(len(detection_boxes)))
        if len(detection_boxes) == 0:
            return [], list(range(len(track_boxes))), []

        cost_matrix = self.build_cost_matrix(track_boxes, detection_boxes)
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = set(range(len(track_boxes)))
        unmatched_detections = set(range(len(detection_boxes)))

        for track_idx, det_idx in zip(track_indices, detection_indices):
            if cost_matrix[track_idx, det_idx] < self.max_distance:
                matches.append((track_idx, det_idx))
                unmatched_tracks.discard(track_idx)
                unmatched_detections.discard(det_idx)

        return matches, list(unmatched_tracks), list(unmatched_detections)
