import pytest
import numpy as np
from pyrobovision.behavior.analyzer import BehaviorAnalyzer
from pyrobovision.behavior.patterns import MotionPattern, BehaviorPattern, extract_motion_features


class TestExtractMotionFeatures:
    def test_empty_positions(self):
        features = extract_motion_features(np.array([]), np.array([]), np.array([]))

        assert features["velocity_magnitude"] == 0.0
        assert features["angular_velocity"] == 0.0
        assert features["acceleration_magnitude"] == 0.0

    def test_stationary_object(self):
        positions = np.array([[0, 0], [0, 0], [0, 0]])
        velocities = np.array([[0, 0], [0, 0], [0, 0]])
        accelerations = np.array([[0, 0], [0, 0]])

        features = extract_motion_features(positions, velocities, accelerations)

        assert features["velocity_magnitude"] == 0.0
        assert features["angular_velocity"] == 0.0

    def test_constant_velocity(self):
        positions = np.array([[0, 0], [1, 0], [2, 0]])
        velocities = np.array([[1, 0], [1, 0], [1, 0]])
        accelerations = np.array([[0, 0], [0, 0]])

        features = extract_motion_features(positions, velocities, accelerations)

        assert np.isclose(features["velocity_magnitude"], 1.0)
        assert features["speed_variance"] == 0.0

    def test_turning_motion(self):
        positions = np.array([[0, 0], [1, 0], [1, 1]])
        velocities = np.array([[1, 0], [0, 1], [0, 1]])
        accelerations = np.array([[-1, 1], [0, 0]])

        features = extract_motion_features(positions, velocities, accelerations)

        assert features["velocity_magnitude"] > 0
        assert features["angular_velocity"] > 0


class TestBehaviorAnalyzer:
    def test_initialization(self):
        analyzer = BehaviorAnalyzer(velocity_threshold=1.0)
        assert analyzer.velocity_threshold == 1.0
        assert len(analyzer.track_states) == 0

    def test_classify_stopped(self):
        analyzer = BehaviorAnalyzer(velocity_threshold=1.0)
        features = {
            "velocity_magnitude": 0.5,
            "angular_velocity": 0.0,
            "acceleration_magnitude": 0.0,
            "speed_variance": 0.0,
            "direction_variance": 0.0,
            "heading_change": 0.0,
        }

        pattern, confidence = analyzer.classify_motion_pattern(features)

        assert pattern == MotionPattern.STOPPED
        assert confidence > 0.9

    def test_classify_moving_straight(self):
        analyzer = BehaviorAnalyzer(velocity_threshold=1.0)
        features = {
            "velocity_magnitude": 5.0,
            "angular_velocity": 0.01,
            "acceleration_magnitude": 0.1,
            "speed_variance": 0.1,
            "direction_variance": 0.01,
            "heading_change": 0.0,
        }

        pattern, confidence = analyzer.classify_motion_pattern(features)

        assert pattern == MotionPattern.MOVING_STRAIGHT
        assert confidence > 0.7

    def test_classify_turning(self):
        analyzer = BehaviorAnalyzer(angular_velocity_threshold=0.1)
        features = {
            "velocity_magnitude": 5.0,
            "angular_velocity": 0.3,
            "acceleration_magnitude": 0.1,
            "speed_variance": 0.1,
            "direction_variance": 0.5,
            "heading_change": 0.2,
        }

        pattern, confidence = analyzer.classify_motion_pattern(features)

        assert pattern == MotionPattern.TURNING
        assert confidence > 0.5

    def test_classify_accelerating(self):
        analyzer = BehaviorAnalyzer(acceleration_threshold=0.5)
        features = {
            "velocity_magnitude": 5.0,
            "angular_velocity": 0.05,
            "acceleration_magnitude": 1.0,
            "speed_variance": 0.5,
            "direction_variance": 0.01,
            "heading_change": 0.0,
        }

        pattern, confidence = analyzer.classify_motion_pattern(features)

        assert pattern == MotionPattern.ACCELERATING
        assert confidence > 0.5

    def test_classify_decelerating(self):
        analyzer = BehaviorAnalyzer(acceleration_threshold=0.5)
        features = {
            "velocity_magnitude": 5.0,
            "angular_velocity": 0.05,
            "acceleration_magnitude": -1.0,
            "speed_variance": 0.5,
            "direction_variance": 0.01,
            "heading_change": 0.0,
        }

        pattern, confidence = analyzer.classify_motion_pattern(features)

        assert pattern == MotionPattern.DECELERATING
        assert confidence > 0.5

    def test_classify_behavior_stationary(self):
        analyzer = BehaviorAnalyzer()
        behavior = analyzer.classify_behavior_pattern(MotionPattern.STOPPED, [])

        assert behavior == BehaviorPattern.STATIONARY

    def test_classify_behavior_linear(self):
        analyzer = BehaviorAnalyzer()
        features = {
            "velocity_magnitude": 5.0,
            "angular_velocity": 0.01,
            "acceleration_magnitude": 0.1,
            "speed_variance": 0.1,
            "direction_variance": 0.01,
            "heading_change": 0.0,
        }

        behavior = analyzer.classify_behavior_pattern(MotionPattern.MOVING_STRAIGHT, [features])

        assert behavior == BehaviorPattern.LINEAR_MOTION

    def test_update_new_track(self):
        analyzer = BehaviorAnalyzer()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])

        state = analyzer.update(1, positions, velocities, accelerations, frame_id=0)

        assert state.track_id == 1
        assert state.frame_count == 1
        assert 1 in analyzer.track_states

    def test_update_existing_track(self):
        analyzer = BehaviorAnalyzer()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])

        analyzer.update(1, positions, velocities, accelerations, frame_id=0)
        state = analyzer.update(1, positions, velocities, accelerations, frame_id=1)

        assert state.frame_count == 2

    def test_get_state(self):
        analyzer = BehaviorAnalyzer()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])

        analyzer.update(1, positions, velocities, accelerations, frame_id=0)
        state = analyzer.get_state(1)

        assert state is not None
        assert state.track_id == 1

    def test_get_nonexistent_state(self):
        analyzer = BehaviorAnalyzer()

        state = analyzer.get_state(999)

        assert state is None

    def test_remove_track(self):
        analyzer = BehaviorAnalyzer()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])

        analyzer.update(1, positions, velocities, accelerations, frame_id=0)
        analyzer.remove_track(1)

        assert 1 not in analyzer.track_states

    def test_detect_lane_change(self):
        analyzer = BehaviorAnalyzer()
        positions = np.array([[0, 0], [1, 0], [1.5, 1]])
        velocities = np.array([[1, 0], [0.5, 1], [0, 1]])
        accelerations = np.array([[-0.5, 1], [-0.5, 0]])

        for i in range(15):
            analyzer.update(1, positions, velocities, accelerations, frame_id=i)

        is_lane_change = analyzer.detect_lane_change(1, history_len=10)

        assert isinstance(is_lane_change, bool)

    def test_get_motion_pattern_duration(self):
        analyzer = BehaviorAnalyzer()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])

        analyzer.update(1, positions, velocities, accelerations, frame_id=0)
        duration = analyzer.get_motion_pattern_duration(1)

        assert duration >= 0

    def test_reset(self):
        analyzer = BehaviorAnalyzer()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])

        analyzer.update(1, positions, velocities, accelerations, frame_id=0)
        analyzer.reset()

        assert len(analyzer.track_states) == 0

    def test_collision_course(self):
        analyzer = BehaviorAnalyzer()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])

        analyzer.update(1, positions, velocities, accelerations, frame_id=0)
        analyzer.update(2, positions + np.array([5, 0]), velocities, accelerations, frame_id=0)

        is_collision = analyzer.is_collision_course(1, 2)

        assert isinstance(is_collision, bool)
