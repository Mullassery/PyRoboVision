"""Occlusion-handling tests for MOTTracker.

CLAUDE.md documents the Kalman-filter-based occlusion behavior as:

    3. Occluded: No detection for N frames (still tracked via prediction)
    4. Death: Track age > threshold without detection -> remove

Prior to this test file, that behavior was documented but never actually
tested — these tests simulate missing detections for N frames and verify
the tracker predicts through the gap, keeps the track's identity, and only
kills the track once the occlusion genuinely exceeds max_age.
"""

import numpy as np
import pytest

from pyrobovision.tracking.mot import MOTTracker, Detection


def _confirm_track(tracker: MOTTracker, bbox: np.ndarray, num_frames: int = 3):
    """Feed `num_frames` identical detections so the track becomes confirmed
    (hit_streak >= 3), then return its track_id."""
    confirmed = []
    for _ in range(num_frames):
        confirmed = tracker.update([Detection(bbox=bbox.copy(), confidence=0.9)])
    assert len(confirmed) == 1, "track should be confirmed after 3 hits"
    return confirmed[0].track_id


class TestOcclusionSurvival:
    def test_track_survives_short_occlusion_gap(self):
        """A track with no matching detection for a few frames (well under
        max_age) must remain alive — not be deleted."""
        tracker = MOTTracker(max_age=10, min_hits=3)
        bbox = np.array([100.0, 100.0, 140.0, 180.0])
        track_id = _confirm_track(tracker, bbox)

        occlusion_frames = 4
        for _ in range(occlusion_frames):
            tracker.update([])  # no detections this frame: object is occluded

        track = tracker.get_track_by_id(track_id)
        assert track is not None, "occluded track was dropped before max_age"
        assert track.is_alive(tracker.max_age)
        assert track.time_since_update == occlusion_frames

    def test_track_is_predicted_forward_during_occlusion(self):
        """While occluded, the Kalman filter should keep advancing the
        track's estimated position using its last known velocity — i.e. the
        tracker is genuinely predicting through the gap, not freezing."""
        tracker = MOTTracker(max_age=10, min_hits=3)

        # Establish a track with clear rightward motion (10px/frame).
        start_x = 100.0
        for i in range(3):
            bbox = np.array([start_x + 10 * i, 100.0, start_x + 40 + 10 * i, 180.0])
            confirmed = tracker.update([Detection(bbox=bbox, confidence=0.9)])
        track_id = confirmed[0].track_id
        track = tracker.get_track_by_id(track_id)

        position_at_occlusion_start = track.get_position().copy()
        velocity = track.get_velocity().copy()
        assert velocity[0] > 0, "track should have picked up rightward velocity"

        # Occlusion: no detections for 3 frames. The Kalman filter's
        # constant-velocity prediction should keep moving x forward.
        for _ in range(3):
            tracker.update([])

        position_after_occlusion = track.get_position()
        assert position_after_occlusion[0] > position_at_occlusion_start[0], (
            "track position did not advance during occlusion — "
            "prediction-through-gap is not working"
        )

    def test_track_remains_confirmed_and_reportable_during_occlusion(self):
        """CLAUDE.md's occlusion claim implies the track is still usable
        (`is_confirmed`) while occluded, e.g. for downstream trajectory
        prediction."""
        tracker = MOTTracker(max_age=10, min_hits=3)
        bbox = np.array([100.0, 100.0, 140.0, 180.0])
        track_id = _confirm_track(tracker, bbox)

        for _ in range(5):
            tracker.update([])

        track = tracker.get_track_by_id(track_id)
        assert track.is_confirmed is True
        assert track in tracker.get_confirmed_tracks()


class TestReacquisitionAfterOcclusion:
    def test_reacquires_same_track_id_after_brief_occlusion(self):
        """When the detector 'sees' the object again in roughly the same
        place after a brief gap, it should re-associate to the SAME track
        (identity preserved), not spawn a new one."""
        tracker = MOTTracker(max_age=10, min_hits=3, iou_threshold=0.3)
        bbox = np.array([100.0, 100.0, 140.0, 180.0])
        track_id = _confirm_track(tracker, bbox)

        # Occluded for 3 frames — no detections.
        for _ in range(3):
            tracker.update([])

        track_before = tracker.get_track_by_id(track_id)
        assert track_before.time_since_update == 3

        # Object reappears close to where it was last seen (overlapping bbox).
        reappear_bbox = np.array([102.0, 101.0, 142.0, 181.0])
        confirmed = tracker.update([Detection(bbox=reappear_bbox, confidence=0.9)])

        assert len(confirmed) == 1
        assert confirmed[0].track_id == track_id, (
            "occluded object was assigned a NEW track id instead of "
            "re-associating with its original track"
        )
        assert confirmed[0].time_since_update == 0
        assert len(tracker.get_all_tracks()) == 1, "a duplicate track was created"

    def test_time_since_update_resets_after_reacquisition(self):
        tracker = MOTTracker(max_age=10, min_hits=3)
        bbox = np.array([100.0, 100.0, 140.0, 180.0])
        track_id = _confirm_track(tracker, bbox)

        for _ in range(4):
            tracker.update([])
        assert tracker.get_track_by_id(track_id).time_since_update == 4

        tracker.update([Detection(bbox=bbox.copy(), confidence=0.9)])
        assert tracker.get_track_by_id(track_id).time_since_update == 0

    def test_occlusion_beyond_max_age_kills_track_and_next_sighting_is_new_id(self):
        """Negative case: if the gap exceeds max_age, the track must die —
        and a detection that shows up afterwards must be treated as a brand
        new object (new track id), not resurrect the old one."""
        tracker = MOTTracker(max_age=5, min_hits=3)
        bbox = np.array([100.0, 100.0, 140.0, 180.0])
        old_track_id = _confirm_track(tracker, bbox)

        for _ in range(6):  # exceeds max_age=5
            tracker.update([])
        assert tracker.get_track_by_id(old_track_id) is None
        assert len(tracker.get_all_tracks()) == 0

        confirmed = tracker.update([Detection(bbox=bbox.copy(), confidence=0.9)])
        assert len(tracker.get_all_tracks()) == 1
        new_track = tracker.get_all_tracks()[0]
        assert new_track.track_id != old_track_id
        assert new_track.time_since_update == 0


class TestMultiObjectOcclusion:
    def test_one_object_occluded_does_not_disturb_other_tracked_object(self):
        """A common real scenario: two tracked objects, one gets briefly
        occluded while the other keeps being detected normally. The visible
        object's track must be unaffected, and the occluded one must survive
        and reacquire independently."""
        tracker = MOTTracker(max_age=10, min_hits=3)
        bbox_a = np.array([0.0, 0.0, 40.0, 40.0])
        bbox_b = np.array([500.0, 500.0, 540.0, 540.0])

        for _ in range(3):
            confirmed = tracker.update(
                [Detection(bbox=bbox_a.copy(), confidence=0.9), Detection(bbox=bbox_b.copy(), confidence=0.9)]
            )
        assert len(confirmed) == 2
        id_a = next(t.track_id for t in confirmed if np.allclose(t.get_bbox(), bbox_a, atol=1.0))
        id_b = next(t.track_id for t in confirmed if t.track_id != id_a)

        # Object B is occluded for 3 frames; object A keeps being detected normally.
        for _ in range(3):
            tracker.update([Detection(bbox=bbox_a.copy(), confidence=0.9)])

        track_a = tracker.get_track_by_id(id_a)
        track_b = tracker.get_track_by_id(id_b)
        assert track_a.time_since_update == 0, "visible track should update every frame"
        assert track_b.time_since_update == 3, "occluded track should accumulate misses"
        assert track_b.is_alive(tracker.max_age)

        # B reappears; both should be confirmed and correctly identified.
        confirmed = tracker.update(
            [Detection(bbox=bbox_a.copy(), confidence=0.9), Detection(bbox=bbox_b.copy(), confidence=0.9)]
        )
        confirmed_ids = {t.track_id for t in confirmed}
        assert confirmed_ids == {id_a, id_b}
