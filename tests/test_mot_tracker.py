import pytest
import numpy as np
from pyrobovision.tracking.mot import MOTTracker, Track, Detection


class TestTrack:
    def test_track_initialization(self):
        track = Track(track_id=1, start_frame=0)
        assert track.track_id == 1
        assert track.age == 1
        assert track.time_since_update == 0
        assert track.is_confirmed is False

    def test_track_update(self):
        track = Track(track_id=1, start_frame=0)
        detection = Detection(bbox=np.array([0, 0, 10, 10]), confidence=0.95)

        track.update(detection, frame_id=1)

        assert len(track.detections) == 1
        assert track.time_since_update == 0
        assert track.hit_streak == 2

    def test_track_confirmation(self):
        track = Track(track_id=1, start_frame=0)
        detection = Detection(bbox=np.array([0, 0, 10, 10]), confidence=0.95)

        assert track.is_confirmed is False

        track.update(detection, 1)
        track.update(detection, 2)
        track.update(detection, 3)

        assert track.is_confirmed is True

    def test_track_predict(self):
        track = Track(track_id=1, start_frame=0)
        detection = Detection(bbox=np.array([0, 0, 10, 10]), confidence=0.95)
        track.kalman_filter.initialize(np.array([5, 5]), vx=1.0, vy=0.0)

        state_before = track.kalman_filter.x.copy()
        track.predict(dt=1.0)
        state_after = track.kalman_filter.x.copy()

        assert state_after[0, 0] > state_before[0, 0]

    def test_track_get_position(self):
        track = Track(track_id=1, start_frame=0)
        track.kalman_filter.initialize(np.array([10, 20]))

        pos = track.get_position()
        assert pos[0] == 10
        assert pos[1] == 20

    def test_track_get_velocity(self):
        track = Track(track_id=1, start_frame=0)
        track.kalman_filter.initialize(np.array([0, 0]), vx=2.0, vy=3.0)

        vel = track.get_velocity()
        assert vel[0] == 2.0
        assert vel[1] == 3.0

    def test_track_is_alive(self):
        track = Track(track_id=1, start_frame=0)
        assert track.is_alive(max_age=30)

        track.time_since_update = 31
        assert not track.is_alive(max_age=30)

    def test_track_get_bbox_from_detection(self):
        track = Track(track_id=1, start_frame=0)
        bbox = np.array([10, 20, 30, 40])
        detection = Detection(bbox=bbox, confidence=0.95)

        track.detections.append((0, detection))

        retrieved_bbox = track.get_bbox()
        assert np.allclose(retrieved_bbox, bbox)


class TestMOTTracker:
    def test_mot_initialization(self):
        tracker = MOTTracker(max_age=30, min_hits=3)
        assert tracker.max_age == 30
        assert len(tracker.tracks) == 0
        assert tracker.frame_id == 0

    def test_mot_single_detection(self):
        tracker = MOTTracker()
        bbox = np.array([10, 20, 30, 40])
        detection = Detection(bbox=bbox, confidence=0.95)

        confirmed = tracker.update([detection])

        assert len(tracker.get_all_tracks()) == 1
        assert len(confirmed) == 0

    def test_mot_confirm_track(self):
        tracker = MOTTracker(min_hits=3)
        bbox = np.array([10, 20, 30, 40])
        detection = Detection(bbox=bbox, confidence=0.95)

        tracker.update([detection])
        tracker.update([detection])
        confirmed = tracker.update([detection])

        assert len(confirmed) == 1
        assert confirmed[0].track_id == 1

    def test_mot_track_persistence(self):
        tracker = MOTTracker()
        bbox1 = np.array([10, 20, 30, 40])
        bbox2 = np.array([12, 22, 32, 42])

        detection1 = Detection(bbox=bbox1, confidence=0.95)
        detection2 = Detection(bbox=bbox2, confidence=0.95)

        tracker.update([detection1])
        tracker.update([detection1])
        tracker.update([detection1])
        confirmed = tracker.update([detection2])

        assert len(confirmed) == 1
        assert confirmed[0].track_id == 1

    def test_mot_multiple_detections(self):
        tracker = MOTTracker()
        detections = [
            Detection(bbox=np.array([10, 20, 30, 40]), confidence=0.95),
            Detection(bbox=np.array([50, 60, 70, 80]), confidence=0.95),
        ]

        tracker.update(detections)
        tracker.update(detections)
        confirmed = tracker.update(detections)

        assert len(confirmed) == 2

    def test_mot_track_removal_on_max_age(self):
        tracker = MOTTracker(max_age=5)
        detection = Detection(bbox=np.array([10, 20, 30, 40]), confidence=0.95)

        tracker.update([detection])
        for _ in range(6):
            tracker.update([])

        assert len(tracker.get_all_tracks()) == 0

    def test_mot_reset(self):
        tracker = MOTTracker()
        detection = Detection(bbox=np.array([10, 20, 30, 40]), confidence=0.95)

        tracker.update([detection])
        tracker.update([detection])
        assert len(tracker.get_all_tracks()) == 1

        tracker.reset()

        assert len(tracker.get_all_tracks()) == 0
        assert tracker.frame_id == 0

    def test_mot_get_track_by_id(self):
        tracker = MOTTracker()
        detection = Detection(bbox=np.array([10, 20, 30, 40]), confidence=0.95)

        tracker.update([detection])
        tracker.update([detection])
        tracker.update([detection])

        track = tracker.get_track_by_id(1)
        assert track is not None
        assert track.track_id == 1

        track_none = tracker.get_track_by_id(999)
        assert track_none is None

    def test_mot_statistics(self):
        tracker = MOTTracker()
        detection = Detection(bbox=np.array([10, 20, 30, 40]), confidence=0.95)

        tracker.update([detection])
        tracker.update([detection])
        tracker.update([detection])

        stats = tracker.get_statistics()
        assert stats["total_tracks"] == 1
        assert stats["confirmed_tracks"] == 1
        assert stats["current_frame"] == 3

    def test_mot_frame_counter(self):
        tracker = MOTTracker()
        detection = Detection(bbox=np.array([10, 20, 30, 40]), confidence=0.95)

        for expected_frame in range(1, 6):
            tracker.update([detection])
            assert tracker.frame_id == expected_frame

    def test_mot_unmatched_detections_create_new_tracks(self):
        tracker = MOTTracker()
        detection1 = Detection(bbox=np.array([10, 20, 30, 40]), confidence=0.95)
        detection2 = Detection(bbox=np.array([50, 60, 70, 80]), confidence=0.95)

        tracker.update([detection1])
        tracker.update([detection1])
        tracker.update([detection1])
        tracker.update([detection2])

        confirmed = tracker.get_confirmed_tracks()
        track_ids = [t.track_id for t in confirmed]
        assert len(confirmed) >= 1

    def test_mot_track_age_increment(self):
        tracker = MOTTracker()
        detection = Detection(bbox=np.array([10, 20, 30, 40]), confidence=0.95)

        tracker.update([detection])
        tracker.update([detection])
        tracker.update([detection])

        track = tracker.get_track_by_id(1)
        assert track.age >= 3

    def test_mot_detection_object_initialization(self):
        bbox = np.array([0, 0, 10, 10])
        det = Detection(bbox=bbox, confidence=0.9, class_id=1)

        assert np.allclose(det.bbox, bbox)
        assert det.confidence == 0.9
        assert det.class_id == 1
        assert det.features is None

    def test_mot_detection_with_features(self):
        bbox = np.array([0, 0, 10, 10])
        features = np.random.randn(128)
        det = Detection(bbox=bbox, confidence=0.9, features=features)

        assert det.features.shape == (128,)
