"""ByteTrack-style association tests with synthetic boxes."""

from __future__ import annotations

import numpy as np

from pyrobovision.tracking import ByteTrackTracker, Detection


def _det(x, y, confidence):
    return Detection(bbox=np.array([x, y, x + 10, y + 10], dtype=float), confidence=confidence)


def test_low_confidence_recovers_existing_track():
    tracker = ByteTrackTracker(min_hits=1)
    for _ in range(3):
        tracker.update([_det(0, 0, 0.9)])
    assert len(tracker.get_all_tracks()) == 1
    tracker.update([_det(1, 0, 0.2)])  # low-confidence, close to the track
    assert len(tracker.get_all_tracks()) == 1
    assert tracker.tracks[0].time_since_update == 0


def test_low_confidence_does_not_create_new_track():
    tracker = ByteTrackTracker(min_hits=1)
    tracker.update([_det(100, 100, 0.2)])
    assert tracker.get_all_tracks() == []


def test_high_confidence_creates_new_track():
    tracker = ByteTrackTracker(min_hits=1)
    tracker.update([_det(100, 100, 0.9)])
    assert len(tracker.get_all_tracks()) == 1
