import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class PointCloud:
    points: np.ndarray
    intensities: Optional[np.ndarray] = None
    reflectance: Optional[np.ndarray] = None
    timestamp: Optional[float] = None

    def __len__(self) -> int:
        return len(self.points)

    def filter_by_range(self, min_range: float, max_range: float) -> "PointCloud":
        distances = np.linalg.norm(self.points, axis=1)
        mask = (distances >= min_range) & (distances <= max_range)

        filtered_points = self.points[mask]
        filtered_intensities = self.intensities[mask] if self.intensities is not None else None
        filtered_reflectance = self.reflectance[mask] if self.reflectance is not None else None

        return PointCloud(
            points=filtered_points,
            intensities=filtered_intensities,
            reflectance=filtered_reflectance,
            timestamp=self.timestamp,
        )

    def filter_by_height(self, min_height: float, max_height: float) -> "PointCloud":
        mask = (self.points[:, 2] >= min_height) & (self.points[:, 2] <= max_height)

        filtered_points = self.points[mask]
        filtered_intensities = self.intensities[mask] if self.intensities is not None else None
        filtered_reflectance = self.reflectance[mask] if self.reflectance is not None else None

        return PointCloud(
            points=filtered_points,
            intensities=filtered_intensities,
            reflectance=filtered_reflectance,
            timestamp=self.timestamp,
        )

    def voxelize(self, voxel_size: float) -> "PointCloud":
        voxel_coords = np.floor(self.points / voxel_size).astype(int)
        unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)

        voxelized_points = []
        voxelized_intensities = [] if self.intensities is not None else None
        voxelized_reflectance = [] if self.reflectance is not None else None

        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            voxelized_points.append(np.mean(self.points[mask], axis=0))

            if self.intensities is not None:
                voxelized_intensities.append(np.mean(self.intensities[mask]))

            if self.reflectance is not None:
                voxelized_reflectance.append(np.mean(self.reflectance[mask]))

        return PointCloud(
            points=np.array(voxelized_points),
            intensities=np.array(voxelized_intensities) if voxelized_intensities else None,
            reflectance=np.array(voxelized_reflectance) if voxelized_reflectance else None,
            timestamp=self.timestamp,
        )

    def downsample(self, num_points: int) -> "PointCloud":
        if len(self.points) <= num_points:
            return self

        indices = np.random.choice(len(self.points), num_points, replace=False)

        downsampled_points = self.points[indices]
        downsampled_intensities = self.intensities[indices] if self.intensities is not None else None
        downsampled_reflectance = self.reflectance[indices] if self.reflectance is not None else None

        return PointCloud(
            points=downsampled_points,
            intensities=downsampled_intensities,
            reflectance=downsampled_reflectance,
            timestamp=self.timestamp,
        )

    def get_statistics(self) -> dict:
        return {
            "num_points": len(self.points),
            "center": np.mean(self.points, axis=0),
            "min": np.min(self.points, axis=0),
            "max": np.max(self.points, axis=0),
            "range": np.linalg.norm(self.points, axis=1).max(),
        }


class LiDARProcessor:
    def __init__(
        self,
        min_range: float = 0.5,
        max_range: float = 100.0,
        min_height: float = -1.0,
        max_height: float = 5.0,
    ):
        self.min_range = min_range
        self.max_range = max_range
        self.min_height = min_height
        self.max_height = max_height

    def process_raw(self, raw_points: np.ndarray) -> PointCloud:
        cloud = PointCloud(points=raw_points.copy())

        cloud = cloud.filter_by_range(self.min_range, self.max_range)
        cloud = cloud.filter_by_height(self.min_height, self.max_height)

        return cloud

    def remove_ground(self, cloud: PointCloud, ground_threshold: float = 0.5) -> PointCloud:
        if len(cloud.points) == 0:
            return cloud

        z_values = cloud.points[:, 2]
        ground_mask = z_values < ground_threshold

        non_ground_points = cloud.points[~ground_mask]
        non_ground_intensities = cloud.intensities[~ground_mask] if cloud.intensities is not None else None
        non_ground_reflectance = cloud.reflectance[~ground_mask] if cloud.reflectance is not None else None

        return PointCloud(
            points=non_ground_points,
            intensities=non_ground_intensities,
            reflectance=non_ground_reflectance,
            timestamp=cloud.timestamp,
        )

    def cluster_points(self, cloud: PointCloud, eps: float = 0.5, min_points: int = 5) -> List[PointCloud]:
        if len(cloud.points) == 0:
            return []

        clusters = []
        visited = np.zeros(len(cloud.points), dtype=bool)

        for i in range(len(cloud.points)):
            if visited[i]:
                continue

            neighbors = self._get_neighbors(cloud.points, i, eps)

            if len(neighbors) < min_points:
                continue

            cluster_points = [cloud.points[i]]
            cluster_indices = [i]
            visited[i] = True

            for neighbor_idx in neighbors:
                if visited[neighbor_idx]:
                    continue

                cluster_points.append(cloud.points[neighbor_idx])
                cluster_indices.append(neighbor_idx)
                visited[neighbor_idx] = True

                new_neighbors = self._get_neighbors(cloud.points, neighbor_idx, eps)
                if len(new_neighbors) >= min_points:
                    neighbors.extend(new_neighbors)

            if len(cluster_points) >= min_points:
                cluster_cloud = PointCloud(
                    points=np.array(cluster_points),
                    intensities=cloud.intensities[cluster_indices] if cloud.intensities is not None else None,
                    reflectance=cloud.reflectance[cluster_indices] if cloud.reflectance is not None else None,
                    timestamp=cloud.timestamp,
                )
                clusters.append(cluster_cloud)

        return clusters

    def _get_neighbors(self, points: np.ndarray, idx: int, eps: float) -> List[int]:
        distances = np.linalg.norm(points - points[idx], axis=1)
        neighbors = np.where((distances < eps) & (distances > 0))[0].tolist()
        return neighbors

    def compute_normals(self, cloud: PointCloud, k: int = 10) -> np.ndarray:
        if len(cloud.points) < k:
            return np.zeros_like(cloud.points)

        normals = np.zeros_like(cloud.points)

        for i in range(len(cloud.points)):
            distances = np.linalg.norm(cloud.points - cloud.points[i], axis=1)
            k_nearest = np.argsort(distances)[:k + 1]

            patch = cloud.points[k_nearest] - cloud.points[i]

            if len(patch) >= 3:
                cov = np.cov(patch.T)
                try:
                    eigenvalues, eigenvectors = np.linalg.eig(cov)
                    min_idx = np.argmin(np.abs(eigenvalues))
                    normals[i] = eigenvectors[:, min_idx].real
                except:
                    normals[i] = np.array([0, 0, 1])

        return normals

    def transform_points(self, cloud: PointCloud, transformation_matrix: np.ndarray) -> PointCloud:
        points_homo = np.hstack([cloud.points, np.ones((len(cloud.points), 1))])
        transformed_points = (transformation_matrix @ points_homo.T).T[:, :3]

        return PointCloud(
            points=transformed_points,
            intensities=cloud.intensities,
            reflectance=cloud.reflectance,
            timestamp=cloud.timestamp,
        )
