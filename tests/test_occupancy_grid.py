import pytest
import numpy as np
from pyrobovision.perception.occupancy import OccupancyGrid, OccupancyGridBuilder
from pyrobovision.perception.bbox_3d import BBox3D


class TestOccupancyGrid:
    def test_initialization(self):
        grid = np.zeros((100, 100))
        occ_grid = OccupancyGrid(
            grid=grid,
            resolution=0.1,
            origin=np.array([-5, -5]),
            height=0.0,
            max_height=2.0,
        )

        assert occ_grid.grid.shape == (100, 100)
        assert occ_grid.resolution == 0.1

    def test_world_to_grid(self):
        grid = np.zeros((100, 100))
        occ_grid = OccupancyGrid(
            grid=grid,
            resolution=0.1,
            origin=np.array([-5, -5]),
            height=0.0,
            max_height=2.0,
        )

        grid_x, grid_y = occ_grid.world_to_grid(0, 0)
        assert grid_x == 50
        assert grid_y == 50

    def test_grid_to_world(self):
        grid = np.zeros((100, 100))
        occ_grid = OccupancyGrid(
            grid=grid,
            resolution=0.1,
            origin=np.array([-5, -5]),
            height=0.0,
            max_height=2.0,
        )

        x, y = occ_grid.grid_to_world(50, 50)
        assert np.isclose(x, 0.0)
        assert np.isclose(y, 0.0)

    def test_set_occupied(self):
        grid = np.zeros((100, 100))
        occ_grid = OccupancyGrid(
            grid=grid,
            resolution=0.1,
            origin=np.array([-5, -5]),
            height=0.0,
            max_height=2.0,
        )

        occ_grid.set_occupied(0, 0, value=1.0)

        assert occ_grid.grid[50, 50] == 1.0

    def test_get_occupancy(self):
        grid = np.zeros((100, 100))
        occ_grid = OccupancyGrid(
            grid=grid,
            resolution=0.1,
            origin=np.array([-5, -5]),
            height=0.0,
            max_height=2.0,
        )

        occ_grid.set_occupied(0, 0, value=1.0)

        occupancy = occ_grid.get_occupancy(0, 0)
        assert occupancy == 1.0

    def test_get_occupied_cells(self):
        grid = np.zeros((100, 100))
        occ_grid = OccupancyGrid(
            grid=grid,
            resolution=0.1,
            origin=np.array([-5, -5]),
            height=0.0,
            max_height=2.0,
        )

        occ_grid.set_occupied(0, 0, value=1.0)
        occ_grid.set_occupied(1, 1, value=0.8)

        occupied = occ_grid.get_occupied_cells(threshold=0.5)
        assert len(occupied) >= 1

    def test_dilate(self):
        grid = np.zeros((100, 100))
        grid[50, 50] = 1.0

        occ_grid = OccupancyGrid(
            grid=grid,
            resolution=0.1,
            origin=np.array([-5, -5]),
            height=0.0,
            max_height=2.0,
        )

        occ_grid.dilate(kernel_size=3)

        assert occ_grid.grid[49, 50] == 1.0
        assert occ_grid.grid[51, 50] == 1.0

    def test_erode(self):
        grid = np.ones((100, 100))

        occ_grid = OccupancyGrid(
            grid=grid,
            resolution=0.1,
            origin=np.array([-5, -5]),
            height=0.0,
            max_height=2.0,
        )

        occ_grid.erode(kernel_size=3)

        assert occ_grid.grid.max() <= 1.0

    def test_get_connectivity_map(self):
        grid = np.zeros((100, 100))
        grid[50:53, 50:53] = 1.0

        occ_grid = OccupancyGrid(
            grid=grid,
            resolution=0.1,
            origin=np.array([-5, -5]),
            height=0.0,
            max_height=2.0,
        )

        connectivity = occ_grid.get_connectivity_map()

        assert connectivity.shape == grid.shape
        assert connectivity[51, 51] > 0


class TestOccupancyGridBuilder:
    def test_initialization(self):
        builder = OccupancyGridBuilder(
            grid_size=(100, 100),
            resolution=0.1,
        )

        assert builder.grid_size == (100, 100)
        assert builder.resolution == 0.1

    def test_from_point_cloud_empty(self):
        builder = OccupancyGridBuilder()

        points = np.array([])

        occ_grid = builder.from_point_cloud(points)

        assert occ_grid.grid.shape == builder.grid_shape

    def test_from_point_cloud_valid(self):
        builder = OccupancyGridBuilder(
            grid_size=(20, 20),
            resolution=0.1,
        )

        points = np.array([
            [0, 0, 0.5],
            [1, 1, 0.5],
            [2, 2, 0.5],
        ])

        occ_grid = builder.from_point_cloud(points)

        occupied = occ_grid.get_occupied_cells(threshold=0.5)
        assert len(occupied) > 0

    def test_from_3d_bboxes(self):
        builder = OccupancyGridBuilder()

        bbox = BBox3D(
            center=np.array([0, 0, 1]),
            dimensions=np.array([1, 1, 1]),
            rotation=np.eye(3),
            confidence=0.9,
        )

        occ_grid = builder.from_3d_bboxes([bbox])

        assert occ_grid.grid.shape == builder.grid_shape

    def test_merge_grids(self):
        builder = OccupancyGridBuilder()

        grid1 = builder.from_point_cloud(np.array([[0, 0, 0.5]]))
        grid2 = builder.from_point_cloud(np.array([[1, 1, 0.5]]))

        merged = builder.merge_grids(grid1, grid2)

        assert merged.grid.shape == grid1.grid.shape
