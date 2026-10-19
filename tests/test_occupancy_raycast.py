"""Raycasting and line-of-sight on the existing OccupancyGrid."""

from __future__ import annotations

import numpy as np
import pytest

from pyrobovision.perception.occupancy import OccupancyGrid


@pytest.fixture
def grid_with_wall() -> OccupancyGrid:
    grid = np.zeros((100, 100))
    occ = OccupancyGrid(
        grid=grid,
        resolution=0.1,
        origin=np.array([-5.0, -5.0]),
        height=0.0,
        max_height=2.0,
    )
    # Wall at x ~= 0, spanning y in [-1, 1].
    for y in np.arange(-1.0, 1.01, 0.1):
        occ.set_occupied(0.0, float(y))
    return occ


def test_raycast_hits_wall(grid_with_wall):
    hit = grid_with_wall.raycast((-3.0, 0.0), (1.0, 0.0), max_range=10.0)
    assert hit is not None
    x, y, distance = hit
    assert x == pytest.approx(0.0, abs=0.11)
    assert y == pytest.approx(0.0, abs=0.11)
    assert distance == pytest.approx(3.0, abs=0.11)


def test_raycast_clear_path_returns_none(grid_with_wall):
    assert grid_with_wall.raycast((-3.0, 3.0), (1.0, 0.0), max_range=10.0) is None


def test_raycast_respects_max_range(grid_with_wall):
    assert grid_with_wall.raycast((-3.0, 0.0), (1.0, 0.0), max_range=1.0) is None


def test_raycast_zero_direction_returns_none(grid_with_wall):
    assert grid_with_wall.raycast((0.0, 0.0), (0.0, 0.0)) is None


def test_line_of_sight_blocked_and_clear(grid_with_wall):
    assert grid_with_wall.line_of_sight((-3.0, 0.0), (3.0, 0.0)) is False
    assert grid_with_wall.line_of_sight((-3.0, 3.0), (3.0, 3.0)) is True
