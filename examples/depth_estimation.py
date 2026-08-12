"""Monocular depth estimation: placeholder heuristic vs. real MiDaS.

Runs standalone with just `python depth_estimation.py` using the default
`model="heuristic"` backend (no extra dependencies, but NOT a real depth model
— see perception/depth.py's module docstring). Pass `--midas` to also run
real MiDaS_small inference, which requires:

    pip install "pyrobovision[depth]"

and a one-time network download of pretrained weights (~90MB) on first use.
"""

import argparse

import numpy as np

from pyrobovision.perception.depth import DepthEstimator
from pyrobovision.perception.bbox_3d import Box3DConverter
from pyrobovision.perception.occupancy import OccupancyGridBuilder


def run_pipeline(model: str) -> None:
    print(f"--- DepthEstimator(model={model!r}) ---")
    estimator = DepthEstimator(model=model)
    estimator.set_calibration(fx=500, fy=500, cx=320, cy=240)

    rgb_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    depth_map = estimator.estimate_depth(rgb_frame)
    print(f"  depth map shape={depth_map.data.shape}, "
          f"min={depth_map.data.min():.2f}, max={depth_map.data.max():.2f}, "
          f"std={depth_map.data.std():.2f}")

    converter = Box3DConverter()
    bbox_3d = converter.from_2d_bbox_and_depth(
        bbox_2d=np.array([100, 80, 200, 180]),
        depth_map=depth_map.data,
        fx=500, fy=500, cx=320, cy=240,
    )
    print(f"  3D box center={bbox_3d.center}, dimensions={bbox_3d.dimensions}")

    builder = OccupancyGridBuilder(grid_size=(100, 100), resolution=0.1)
    occupancy_grid = builder.from_3d_bboxes([bbox_3d])
    print(f"  occupied cells: {len(occupancy_grid.get_occupied_cells())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--midas", action="store_true",
                         help="also run real MiDaS inference (requires pyrobovision[depth])")
    args = parser.parse_args()

    run_pipeline("heuristic")

    if args.midas:
        print()
        run_pipeline("midas")
    else:
        print("\n(pass --midas to also run real MiDaS inference, "
              "requires: pip install \"pyrobovision[depth]\")")


if __name__ == "__main__":
    main()
