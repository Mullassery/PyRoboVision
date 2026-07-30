import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BBox3D:
    center: np.ndarray
    dimensions: np.ndarray
    rotation: np.ndarray
    confidence: float
    class_id: int = 0

    def volume(self) -> float:
        return float(np.prod(self.dimensions))

    def to_corners(self) -> np.ndarray:
        dx, dy, dz = self.dimensions / 2.0

        corners_local = np.array([
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, dz],
            [-dx, dy, dz],
        ])

        corners_world = corners_local @ self.rotation.T + self.center
        return corners_world

    def get_projection_2d(self, intrinsics: np.ndarray) -> np.ndarray:
        corners_3d = self.to_corners()
        corners_homo = corners_3d @ intrinsics.T
        corners_2d = corners_homo[:, :2] / corners_homo[:, 2:3]
        return corners_2d

    def intersection_over_union(self, other: "BBox3D") -> float:
        self_volume = self.volume()
        other_volume = other.volume()

        if self_volume == 0 or other_volume == 0:
            return 0.0

        center_dist = np.linalg.norm(self.center - other.center)
        max_dist = np.linalg.norm(self.dimensions) + np.linalg.norm(other.dimensions)

        if center_dist > max_dist:
            return 0.0

        overlap = 1.0 - (center_dist / max_dist)
        intersection = overlap * min(self_volume, other_volume)
        union = self_volume + other_volume - intersection

        return float(intersection / union) if union > 0 else 0.0


class Box3DConverter:
    def __init__(self, intrinsics: Optional[np.ndarray] = None):
        self.intrinsics = intrinsics or np.eye(3)

    def from_2d_bbox_and_depth(
        self,
        bbox_2d: np.ndarray,
        depth_map: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        class_id: int = 0,
        confidence: float = 0.9,
    ) -> BBox3D:
        x_min, y_min, x_max, y_max = bbox_2d

        center_x_2d = (x_min + x_max) / 2
        center_y_2d = (y_min + y_max) / 2

        center_x_2d = int(np.clip(center_x_2d, 0, depth_map.shape[1] - 1))
        center_y_2d = int(np.clip(center_y_2d, 0, depth_map.shape[0] - 1))

        depth = depth_map[center_y_2d, center_x_2d]

        center_x_3d = (center_x_2d - cx) * depth / fx
        center_y_3d = (center_y_2d - cy) * depth / fy
        center_z_3d = depth

        width_2d = x_max - x_min
        height_2d = y_max - y_min

        width_3d = (width_2d / fx) * depth
        height_3d = (height_2d / fy) * depth
        length_3d = depth * 0.5

        return BBox3D(
            center=np.array([center_x_3d, center_y_3d, center_z_3d]),
            dimensions=np.array([width_3d, height_3d, length_3d]),
            rotation=np.eye(3),
            confidence=confidence,
            class_id=class_id,
        )

    def from_point_cloud(
        self,
        points: np.ndarray,
        class_id: int = 0,
        confidence: float = 0.9,
    ) -> BBox3D:
        if len(points) == 0:
            return BBox3D(
                center=np.zeros(3),
                dimensions=np.ones(3),
                rotation=np.eye(3),
                confidence=0.0,
                class_id=class_id,
            )

        center = np.mean(points, axis=0)

        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords

        pca_matrix = self._compute_pca(points)

        return BBox3D(
            center=center,
            dimensions=dimensions,
            rotation=pca_matrix,
            confidence=confidence,
            class_id=class_id,
        )

    def _compute_pca(self, points: np.ndarray) -> np.ndarray:
        centered_points = points - np.mean(points, axis=0)

        if len(centered_points) < 3:
            return np.eye(3)

        cov_matrix = np.cov(centered_points.T)

        try:
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            sorted_idx = np.argsort(eigenvalues)[::-1]
            return eigenvectors[:, sorted_idx].real.astype(np.float32)
        except:
            return np.eye(3)

    def transform_3d_bbox(self, bbox: BBox3D, transform_matrix: np.ndarray) -> BBox3D:
        center_homo = np.append(bbox.center, 1)
        new_center = (transform_matrix @ center_homo)[:3]

        rotation_part = transform_matrix[:3, :3]
        new_rotation = rotation_part @ bbox.rotation

        return BBox3D(
            center=new_center,
            dimensions=bbox.dimensions,
            rotation=new_rotation,
            confidence=bbox.confidence,
            class_id=bbox.class_id,
        )

    def compute_iou_3d(self, bbox1: BBox3D, bbox2: BBox3D) -> float:
        return bbox1.intersection_over_union(bbox2)
