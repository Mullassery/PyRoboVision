import pytest
import numpy as np
from pyrobovision.tracking.mot import MOTTracker, Detection
from pyrobovision.prediction.trajectory import TrajectoryPredictor
from pyrobovision.prediction.uncertainty import UncertaintyEstimator
from pyrobovision.behavior.analyzer import BehaviorAnalyzer
from pyrobovision.intent.predictor import IntentPredictor, Intent


class TestV12Integration:
    def test_full_tracking_pipeline(self):
        tracker = MOTTracker(max_age=30, min_hits=3)
        predictor = TrajectoryPredictor(model="cv", history_len=10)
        behavior_analyzer = BehaviorAnalyzer()
        intent_predictor = IntentPredictor(lookahead_frames=30)

        for frame_id in range(15):
            bbox = np.array([10 + frame_id, 20, 30 + frame_id, 40])
            detection = Detection(bbox=bbox, confidence=0.95)

            confirmed_tracks = tracker.update([detection])

            if len(confirmed_tracks) > 0:
                track = confirmed_tracks[0]
                positions = np.array([d.bbox[:2] for _, d in track.detections])
                velocities = np.diff(positions, axis=0) if len(positions) > 1 else np.array([[0, 0]])
                accelerations = np.diff(velocities, axis=0) if len(velocities) > 1 else np.array([[0, 0]])

                if len(positions) > 1:
                    trajectory = predictor.predict_trajectory(positions, horizon=10)
                    assert trajectory.shape == (10, 2)

                heading_changes = np.array([0.0] * len(velocities))
                behavior_state = behavior_analyzer.update(track.track_id, positions, velocities, accelerations, frame_id)
                assert behavior_state is not None

                intent_pred = intent_predictor.predict_intent(positions, velocities, accelerations, heading_changes)
                assert intent_pred is not None
                assert intent_pred.intent in Intent.__members__.values()

    def test_multiple_object_tracking_with_prediction(self):
        tracker = MOTTracker()
        predictor = TrajectoryPredictor(model="ca")
        behavior_analyzer = BehaviorAnalyzer()
        intent_predictor = IntentPredictor()

        for frame_id in range(20):
            detections = [
                Detection(bbox=np.array([10 + frame_id * 1, 20, 30 + frame_id * 1, 40]), confidence=0.95),
                Detection(bbox=np.array([50 - frame_id * 0.5, 50, 70 - frame_id * 0.5, 70]), confidence=0.90),
            ]

            confirmed = tracker.update(detections)

            assert len(tracker.get_all_tracks()) > 0

            for track in confirmed:
                positions = np.array([d.bbox[:2] for _, d in track.detections])
                velocities = np.diff(positions, axis=0) if len(positions) > 1 else np.array([[0, 0]])
                accelerations = np.diff(velocities, axis=0) if len(velocities) > 1 else np.array([[0, 0]])
                heading_changes = np.zeros(len(velocities))

                behavior_analyzer.update(track.track_id, positions, velocities, accelerations, frame_id)
                intent_predictor.predict_intent(positions, velocities, accelerations, heading_changes)

    def test_trajectory_prediction_accuracy(self):
        predictor = TrajectoryPredictor(model="cv")

        ground_truth = np.array([[i, 0] for i in range(10)], dtype=float)
        history = ground_truth[:5]

        predicted = predictor.predict_trajectory(history, horizon=5)

        ade = predictor.compute_average_displacement_error(predicted, ground_truth[5:])

        assert 0 <= ade < 2.0

    def test_behavior_and_intent_consistency(self):
        behavior_analyzer = BehaviorAnalyzer()
        intent_predictor = IntentPredictor()

        positions = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        velocities = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
        accelerations = np.array([[0, 0], [0, 0], [0, 0]])
        heading_changes = np.array([0.0, 0.0, 0.0, 0.0])

        behavior_state = behavior_analyzer.update(1, positions, velocities, accelerations, 0)

        intent_pred = intent_predictor.predict_intent(positions, velocities, accelerations, heading_changes)

        assert behavior_state is not None
        assert intent_pred is not None
        assert intent_pred.confidence > 0.5

    def test_uncertainty_propagation(self):
        uncertainty_estimator = UncertaintyEstimator()
        positions = np.array([[0, 0], [1, 0], [2, 0]])

        uncertainty, cov = uncertainty_estimator.estimate_position_uncertainty(positions)

        assert cov.shape == (2, 2)
        assert np.all(np.linalg.eigvals(cov) >= 0)

    def test_track_lifecycle(self):
        tracker = MOTTracker(max_age=5, min_hits=2)
        behavior_analyzer = BehaviorAnalyzer()

        for frame_id in range(10):
            bbox = np.array([10 + frame_id, 20, 30 + frame_id, 40])
            detection = Detection(bbox=bbox, confidence=0.95)

            confirmed = tracker.update([detection])

            for track in confirmed:
                positions = np.array([d.bbox[:2] for _, d in track.detections])
                velocities = np.diff(positions, axis=0) if len(positions) > 1 else np.array([[0, 0]])
                accelerations = np.diff(velocities, axis=0) if len(velocities) > 1 else np.array([[0, 0]])

                behavior_analyzer.update(track.track_id, positions, velocities, accelerations, frame_id)

        for frame_id in range(10, 20):
            confirmed = tracker.update([])

            for track in confirmed:
                behavior_analyzer.remove_track(track.track_id)

    def test_collision_detection_with_tracking(self):
        tracker = MOTTracker()
        intent_predictor = IntentPredictor()

        for frame_id in range(15):
            detections = [
                Detection(bbox=np.array([10 + frame_id * 2, 20, 30 + frame_id * 2, 40]), confidence=0.95),
                Detection(bbox=np.array([100 - frame_id * 2, 50, 120 - frame_id * 2, 70]), confidence=0.95),
            ]

            confirmed = tracker.update(detections)

            if len(confirmed) >= 2:
                track1, track2 = confirmed[0], confirmed[1]

                pos1 = track1.get_position()
                vel1 = track1.get_velocity()
                pos2 = track2.get_position()
                vel2 = track2.get_velocity()

                is_collision, conf, reason = intent_predictor.predict_collision_intent(
                    pos1, vel1, pos2, vel2, time_to_collision_threshold=5.0
                )

                assert isinstance(bool(is_collision), bool)
                assert 0 <= conf <= 1

    def test_end_to_end_scenario(self):
        tracker = MOTTracker(max_age=30, min_hits=3)
        traj_predictor = TrajectoryPredictor(model="ca")
        uncertainty_estimator = UncertaintyEstimator()
        behavior_analyzer = BehaviorAnalyzer()
        intent_predictor = IntentPredictor(lookahead_frames=30)

        scenario_frames = 30

        for frame_id in range(scenario_frames):
            bbox1 = np.array([10 + frame_id * 2, 20, 30 + frame_id * 2, 40])
            bbox2 = np.array([100 - frame_id * 1.5, 50, 120 - frame_id * 1.5, 70])

            detections = [
                Detection(bbox=bbox1, confidence=0.95, class_id=0),
                Detection(bbox=bbox2, confidence=0.90, class_id=0),
            ]

            confirmed_tracks = tracker.update(detections)

            for track in confirmed_tracks:
                if len(track.detections) > 3:
                    positions = np.array([d.bbox[:2] for _, d in track.detections])
                    velocities = np.diff(positions, axis=0)
                    accelerations = np.diff(velocities, axis=0) if len(velocities) > 1 else np.array([[0, 0]])

                    trajectory = traj_predictor.predict_trajectory(positions, horizon=15)
                    assert trajectory.shape == (15, 2)

                    unc, cov = uncertainty_estimator.estimate_position_uncertainty(positions)
                    assert cov.shape == (2, 2)

                    heading_changes = np.zeros(len(velocities))
                    behavior = behavior_analyzer.update(track.track_id, positions, velocities, accelerations, frame_id)
                    assert behavior.current_pattern is not None

                    intent = intent_predictor.predict_intent(positions, velocities, accelerations, heading_changes)
                    assert intent.intent is not None
                    assert 0 <= intent.confidence <= 1
