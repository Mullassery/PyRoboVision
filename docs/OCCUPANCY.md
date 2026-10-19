# Occupancy grids: raycasting and line of sight

`pyrobovision.perception.occupancy.OccupancyGrid` is a 2D BEV grid used for
sensor perception utilities. It now supports deterministic ray traversal for
sensor simulation and visibility checks.

## Grid model

| Field | Meaning |
|---|---|
| `grid` | 2D NumPy array, indexed `[row=y, col=x]` |
| `resolution` | metres per cell |
| `origin` | world `(x, y)` of grid cell `(0, 0)` |
| `height`, `max_height` | vertical metadata carried from builders |

Values `> threshold` (default `0.5`) are treated as occupied.

## `raycast`

```python
hit = grid.raycast(
    origin=(-3.0, 0.0),      # world metres
    direction=(1.0, 0.0),    # need not be normalized
    max_range=10.0,          # optional; defaults to the grid diagonal
    threshold=0.5,
)
# (x, y, distance_m) for the first occupied cell, or None
```

Traversal uses the Amanatides–Woo DDA algorithm, stepping cell-by-cell without
sampling gaps. A ray that begins inside an occupied cell returns distance `0.0`.
A ray that leaves the grid or exceeds `max_range` returns `None`.

## `line_of_sight`

```python
visible = grid.line_of_sight((-3.0, 0.0), (3.0, 0.0))
```

Returns `True` when no occupied cell blocks the segment. It calls `raycast` with
`max_range` equal to the segment length; an occupied target cell counts as
blocked.

## Limits

- 2D BEV only; this is not a 3D voxel grid.
- Static occupancy; no probabilistic updates or sensor noise.
- The occupied threshold is per-call, not stored on the grid.
- `grid_to_world` returns the cell corner; `raycast` returns the exact
  intersection point along the ray.
