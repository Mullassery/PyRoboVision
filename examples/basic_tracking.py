"""Basic multi-object tracking + trajectory prediction + occlusion handling.

Runs standalone (no data files, no GPU, no torch) — just `python basic_tracking.py`.

Feeds a few frames of synthetic bounding boxes for a single object moving right,
simulates it being occluded for a few frames (no detections), and shows the
Kalman filter continuing to predict its position through the gap and
re-associating it to the same track_id once it reappears.
"""

import numpy as np

from pyrobovision.tracking.mot import MOTTracker, Detection
from pyrobovision.prediction.trajectory import TrajectoryPredictor


def main() -> None:
    tracker = MOTTracker(max_age=30, min_hits=3)

    print("--- Frames 1-4: object moving right, detected every frame ---")
    boxes = [
        np.array([100.0, 100.0, 140.0, 180.0]),
        np.array([110.0, 100.0, 150.0, 180.0]),
        np.array([120.0, 100.0, 160.0, 180.0]),
        np.array([130.0, 100.0, 170.0, 180.0]),
    ]
    positions = []
    for i, box in enumerate(boxes, start=1):
        confirmed = tracker.update([Detection(bbox=box, confidence=0.9)])
        track = tracker.get_track_by_id(1)
        positions.append(track.get_position())
        print(f"  frame {i}: track {track.track_id} pos={track.get_position()} "
              f"confirmed={track.is_confirmed}")

    print("\n--- Frames 5-7: object occluded (no detections) ---")
    for i in range(5, 8):
        tracker.update([])  # nothing detected this frame
        track = tracker.get_track_by_id(1)
        print(f"  frame {i}: track {track.track_id} predicted pos={track.get_position()} "
              f"time_since_update={track.time_since_update} confirmed={track.is_confirmed}")

    print("\n--- Frame 8: object reappears near its predicted location ---")
    reappear_box = np.array([170.0, 100.0, 210.0, 180.0])
    confirmed = tracker.update([Detection(bbox=reappear_box, confidence=0.9)])
    print(f"  frame 8: re-associated to track {confirmed[0].track_id} "
          f"(same id as before: {confirmed[0].track_id == 1})")

    print("\n--- Trajectory prediction from the pre-occlusion history ---")
    predictor = TrajectoryPredictor(model="cv")  # "cv" or "ca"
    future = predictor.predict_trajectory(np.array(positions), horizon=5)
    print(f"  predicted next 5 positions:\n{future}")


if __name__ == "__main__":
    main()
