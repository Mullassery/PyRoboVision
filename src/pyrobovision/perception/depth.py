import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DepthMap:
    data: np.ndarray
    scale: float
    shift: float
    min_depth: float
    max_depth: float

    def get_depth(self, x: int, y: int) -> float:
        if 0 <= y < self.data.shape[0] and 0 <= x < self.data.shape[1]:
            return self.data[y, x]
        return self.max_depth

    def get_depth_batch(self, points: np.ndarray) -> np.ndarray:
        depths = []
        for x, y in points:
            depths.append(self.get_depth(int(x), int(y)))
        return np.array(depths)

    def upsample(self, factor: int = 2) -> "DepthMap":
        h, w = self.data.shape
        new_h, new_w = h * factor, w * factor
        upsampled = np.zeros((new_h, new_w))

        for i in range(h):
            for j in range(w):
                for di in range(factor):
                    for dj in range(factor):
                        upsampled[i * factor + di, j * factor + dj] = self.data[i, j]

        return DepthMap(
            data=upsampled,
            scale=self.scale,
            shift=self.shift,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

    def downsample(self, factor: int = 2) -> "DepthMap":
        h, w = self.data.shape
        new_h, new_w = h // factor, w // factor
        downsampled = np.zeros((new_h, new_w))

        for i in range(new_h):
            for j in range(new_w):
                patch = self.data[i * factor : (i + 1) * factor, j * factor : (j + 1) * factor]
                downsampled[i, j] = np.mean(patch)

        return DepthMap(
            data=downsampled,
            scale=self.scale,
            shift=self.shift,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

    def to_3d(self, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        h, w = self.data.shape
        points_3d = []

        for y in range(h):
            for x in range(w):
                z = self.data[y, x]
                if self.min_depth <= z <= self.max_depth:
                    x_3d = (x - cx) * z / fx
                    y_3d = (y - cy) * z / fy
                    points_3d.append([x_3d, y_3d, z])

        return np.array(points_3d) if points_3d else np.zeros((0, 3))


class DepthEstimator:
    def __init__(
        self,
        model: str = "midas",
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        scale: float = 0.001,
    ):
        self.model = model
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scale = scale
        self.calibration_matrix = None

    def set_calibration(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self.calibration_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def estimate_depth(self, image: np.ndarray, use_median: bool = True) -> DepthMap:
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        base_depth = 5.0
        edge_detection = self._detect_edges(image)

        depth_estimate = np.ones((h, w)) * base_depth
        depth_estimate[edge_detection > 0] *= 0.7
        depth_estimate[edge_detection == 0] *= 1.3

        if use_median:
            depth_estimate = self._apply_median_filter(depth_estimate, kernel_size=5)

        depth_map = DepthMap(
            data=np.clip(depth_estimate, self.min_depth, self.max_depth),
            scale=self.scale,
            shift=0.0,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

        return depth_map

    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        h, w = gray.shape
        edges = np.zeros((h, w))

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                patch = gray[i - 1 : i + 2, j - 1 : j + 2]
                edge_x = np.sum(patch * gx)
                edge_y = np.sum(patch * gy)
                edges[i, j] = np.sqrt(edge_x**2 + edge_y**2)

        return (edges > np.percentile(edges, 50)).astype(float)

    def _apply_median_filter(self, depth: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        h, w = depth.shape
        filtered = depth.copy()
        pad = kernel_size // 2

        for i in range(pad, h - pad):
            for j in range(pad, w - pad):
                patch = depth[i - pad : i + pad + 1, j - pad : j + pad + 1]
                filtered[i, j] = np.median(patch)

        return filtered

    def fuse_with_lidar(self, depth_map: DepthMap, lidar_points: np.ndarray, weight: float = 0.5) -> DepthMap:
        fused_depth = depth_map.data.copy()

        if self.calibration_matrix is None:
            return depth_map

        fx = self.calibration_matrix[0, 0]
        fy = self.calibration_matrix[1, 1]
        cx = self.calibration_matrix[0, 2]
        cy = self.calibration_matrix[1, 2]

        for point in lidar_points:
            if len(point) >= 3:
                x_3d, y_3d, z_3d = point[:3]

                if z_3d > 0:
                    x_2d = int((x_3d * fx / z_3d) + cx)
                    y_2d = int((y_3d * fy / z_3d) + cy)

                    if 0 <= y_2d < fused_depth.shape[0] and 0 <= x_2d < fused_depth.shape[1]:
                        current_depth = fused_depth[y_2d, x_2d]
                        fused_depth[y_2d, x_2d] = (1 - weight) * current_depth + weight * z_3d

        fused_map = DepthMap(
            data=np.clip(fused_depth, self.min_depth, self.max_depth),
            scale=depth_map.scale,
            shift=depth_map.shift,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

        return fused_map

    def estimate_uncertainty(self, depth_map: DepthMap) -> np.ndarray:
        h, w = depth_map.data.shape
        uncertainty = np.zeros((h, w))

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                patch = depth_map.data[i - 1 : i + 2, j - 1 : j + 2]
                uncertainty[i, j] = np.std(patch)

        return uncertainty
