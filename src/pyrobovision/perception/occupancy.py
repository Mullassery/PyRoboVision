import math

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class OccupancyGrid:
    grid: np.ndarray
    resolution: float
    origin: np.ndarray
    height: float
    max_height: float

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return x, y

    def set_occupied(self, x: float, y: float, value: float = 1.0) -> None:
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.grid.shape[1] and 0 <= grid_y < self.grid.shape[0]:
            self.grid[grid_y, grid_x] = max(self.grid[grid_y, grid_x], value)

    def set_free(self, x: float, y: float) -> None:
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.grid.shape[1] and 0 <= grid_y < self.grid.shape[0]:
            self.grid[grid_y, grid_x] = 0.0

    def get_occupancy(self, x: float, y: float) -> float:
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.grid.shape[1] and 0 <= grid_y < self.grid.shape[0]:
            return float(self.grid[grid_y, grid_x])
        return 0.0

    def raycast(
        self,
        origin: Tuple[float, float],
        direction: Tuple[float, float],
        max_range: Optional[float] = None,
        threshold: float = 0.5,
    ) -> Optional[Tuple[float, float, float]]:
        """Cast a 2D ray through the grid (Amanatides–Woo traversal).

        Returns ``(x, y, distance_m)`` for the first occupied cell, or ``None``
        if the ray leaves the grid or reaches ``max_range`` without a hit.
        """
        ox, oy = float(origin[0]), float(origin[1])
        dx, dy = float(direction[0]), float(direction[1])
        norm = math.hypot(dx, dy)
        if norm == 0.0:
            return None
        dx, dy = dx / norm, dy / norm

        if max_range is None:
            max_range = math.hypot(
                self.grid.shape[1] * self.resolution,
                self.grid.shape[0] * self.resolution,
            )

        # Continuous grid coordinates; cell (ix, iy) owns [ix, ix+1) x [iy, iy+1).
        gx = (ox - self.origin[0]) / self.resolution
        gy = (oy - self.origin[1]) / self.resolution
        ix, iy = math.floor(gx), math.floor(gy)

        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        delta_x = self.resolution / abs(dx) if dx != 0 else math.inf
        delta_y = self.resolution / abs(dy) if dy != 0 else math.inf

        if dx > 0:
            next_x = (ix + 1) * self.resolution + self.origin[0]
        elif dx < 0:
            next_x = ix * self.resolution + self.origin[0]
        else:
            next_x = math.inf
        if dy > 0:
            next_y = (iy + 1) * self.resolution + self.origin[1]
        elif dy < 0:
            next_y = iy * self.resolution + self.origin[1]
        else:
            next_y = math.inf

        t_max_x = (next_x - ox) / dx if dx != 0 else math.inf
        t_max_y = (next_y - oy) / dy if dy != 0 else math.inf

        def _in_bounds(cx: int, cy: int) -> bool:
            return 0 <= cy < self.grid.shape[0] and 0 <= cx < self.grid.shape[1]

        if _in_bounds(ix, iy) and self.grid[iy, ix] > threshold:
            return ox, oy, 0.0

        distance = 0.0
        while distance <= max_range:
            if t_max_x < t_max_y:
                distance = t_max_x
                ix += step_x
                t_max_x += delta_x
            else:
                distance = t_max_y
                iy += step_y
                t_max_y += delta_y

            if distance > max_range:
                return None
            if not _in_bounds(ix, iy):
                return None
            if self.grid[iy, ix] > threshold:
                return ox + dx * distance, oy + dy * distance, distance
        return None

    def line_of_sight(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        threshold: float = 0.5,
    ) -> bool:
        """Whether no occupied cell blocks the segment ``start`` → ``end``."""
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        distance = math.hypot(dx, dy)
        if distance == 0.0:
            return True
        hit = self.raycast(start, (dx, dy), max_range=distance, threshold=threshold)
        return hit is None

    def get_occupied_cells(self, threshold: float = 0.5) -> List[Tuple[float, float]]:
        occupied = []
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] > threshold:
                    x, y = self.grid_to_world(j, i)
                    occupied.append((x, y))
        return occupied

    def dilate(self, kernel_size: int = 3) -> None:
        h, w = self.grid.shape
        dilated = np.zeros_like(self.grid)

        for i in range(h):
            for j in range(w):
                if self.grid[i, j] > 0:
                    for di in range(-kernel_size // 2, kernel_size // 2 + 1):
                        for dj in range(-kernel_size // 2, kernel_size // 2 + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                dilated[ni, nj] = max(dilated[ni, nj], self.grid[i, j])

        self.grid = dilated

    def erode(self, kernel_size: int = 3) -> None:
        h, w = self.grid.shape
        eroded = np.ones_like(self.grid)

        for i in range(h):
            for j in range(w):
                min_val = 1.0
                for di in range(-kernel_size // 2, kernel_size // 2 + 1):
                    for dj in range(-kernel_size // 2, kernel_size // 2 + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            min_val = min(min_val, self.grid[ni, nj])
                eroded[i, j] = min_val

        self.grid = eroded

    def get_connectivity_map(self) -> np.ndarray:
        occupied = (self.grid > 0.5).astype(int)

        connectivity = np.zeros_like(occupied)
        for i in range(occupied.shape[0]):
            for j in range(occupied.shape[1]):
                if occupied[i, j]:
                    neighbor_count = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < occupied.shape[0] and 0 <= nj < occupied.shape[1]:
                                if occupied[ni, nj]:
                                    neighbor_count += 1
                    connectivity[i, j] = neighbor_count

        return connectivity


class OccupancyGridBuilder:
    def __init__(
        self,
        grid_size: Tuple[float, float] = (100.0, 100.0),
        resolution: float = 0.1,
        origin: Optional[np.ndarray] = None,
        height: float = 0.0,
        max_height: float = 2.0,
    ):
        self.grid_size = grid_size
        self.resolution = resolution
        self.origin = origin or np.array([-grid_size[0] / 2, -grid_size[1] / 2])
        self.height = height
        self.max_height = max_height

        grid_width = int(grid_size[0] / resolution)
        grid_height = int(grid_size[1] / resolution)
        self.grid_shape = (grid_height, grid_width)

    def from_point_cloud(self, points: np.ndarray) -> OccupancyGrid:
        grid = np.zeros(self.grid_shape)

        for point in points:
            if len(point) >= 3:
                x, y, z = point[:3]

                if self.height <= z <= self.max_height:
                    grid_x = int((x - self.origin[0]) / self.resolution)
                    grid_y = int((y - self.origin[1]) / self.resolution)

                    if 0 <= grid_x < self.grid_shape[1] and 0 <= grid_y < self.grid_shape[0]:
                        grid[grid_y, grid_x] = 1.0

        return OccupancyGrid(
            grid=grid,
            resolution=self.resolution,
            origin=self.origin.copy(),
            height=self.height,
            max_height=self.max_height,
        )

    def from_3d_bboxes(self, bboxes: List) -> OccupancyGrid:
        grid = np.zeros(self.grid_shape)

        for bbox in bboxes:
            corners = bbox.to_corners()

            for corner in corners:
                if len(corner) >= 3:
                    x, y, z = corner[:3]

                    if self.height <= z <= self.max_height:
                        grid_x = int((x - self.origin[0]) / self.resolution)
                        grid_y = int((y - self.origin[1]) / self.resolution)

                        if 0 <= grid_x < self.grid_shape[1] and 0 <= grid_y < self.grid_shape[0]:
                            grid[grid_y, grid_x] = 1.0

            center_x = bbox.center[0]
            center_y = bbox.center[1]
            cx = int((center_x - self.origin[0]) / self.resolution)
            cy = int((center_y - self.origin[1]) / self.resolution)

            if 0 <= cx < self.grid_shape[1] and 0 <= cy < self.grid_shape[0]:
                grid[cy, cx] = 1.0

        return OccupancyGrid(
            grid=grid,
            resolution=self.resolution,
            origin=self.origin.copy(),
            height=self.height,
            max_height=self.max_height,
        )

    def from_costmap(self, costmap: np.ndarray, threshold: float = 0.5) -> OccupancyGrid:
        grid = (costmap > threshold).astype(float)

        if grid.shape != self.grid_shape:
            h, w = self.grid_shape
            resized_grid = np.zeros(self.grid_shape)
            h_scale = grid.shape[0] / h
            w_scale = grid.shape[1] / w

            for i in range(h):
                for j in range(w):
                    src_i = int(i * h_scale)
                    src_j = int(j * w_scale)
                    resized_grid[i, j] = grid[min(src_i, grid.shape[0] - 1), min(src_j, grid.shape[1] - 1)]

            grid = resized_grid

        return OccupancyGrid(
            grid=grid,
            resolution=self.resolution,
            origin=self.origin.copy(),
            height=self.height,
            max_height=self.max_height,
        )

    def merge_grids(self, grid1: OccupancyGrid, grid2: OccupancyGrid) -> OccupancyGrid:
        merged_grid = np.maximum(grid1.grid, grid2.grid)

        return OccupancyGrid(
            grid=merged_grid,
            resolution=self.resolution,
            origin=self.origin.copy(),
            height=self.height,
            max_height=self.max_height,
        )
