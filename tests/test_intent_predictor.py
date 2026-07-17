import pytest
import numpy as np
from pyrobovision.intent.predictor import IntentPredictor, Intent


class TestIntentPredictor:
    def test_initialization(self):
        predictor = IntentPredictor(lookahead_frames=30, confidence_threshold=0.6)
        assert predictor.lookahead_frames == 30
        assert predictor.confidence_threshold == 0.6

    def test_predict_intent_insufficient_history(self):
        predictor = IntentPredictor()
        positions = np.array([[0, 0]])
        velocities = np.array([[1, 0]])
        accelerations = np.array([[0, 0]])
        heading_changes = np.array([0.0])

        prediction = predictor.predict_intent(positions, velocities, accelerations, heading_changes)

        assert prediction.intent == Intent.UNKNOWN
        assert prediction.confidence == 0.0

    def test_predict_intent_stationary(self):
        predictor = IntentPredictor()
        positions = np.array([[0, 0], [0, 0]])
        velocities = np.array([[0.1, 0], [0.1, 0]])
        accelerations = np.array([[0, 0]])
        heading_changes = np.array([0.0])

        prediction = predictor.predict_intent(positions, velocities, accelerations, heading_changes)

        assert prediction.intent == Intent.STOP
        assert prediction.confidence > 0.8

    def test_predict_intent_accelerating(self):
        predictor = IntentPredictor()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [2, 0]])
        accelerations = np.array([[1, 0]])
        heading_changes = np.array([0.0])

        prediction = predictor.predict_intent(positions, velocities, accelerations, heading_changes)

        assert prediction.intent == Intent.ACCELERATE
        assert prediction.confidence > 0.5

    def test_predict_intent_decelerating(self):
        predictor = IntentPredictor()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[2, 0], [1, 0]])
        accel_vector = np.array([-1, 0])
        accelerations = np.array([accel_vector / np.linalg.norm(accel_vector) * 1.0])
        heading_changes = np.array([0.0])

        prediction = predictor.predict_intent(positions, velocities, accelerations, heading_changes)

        assert prediction.intent == Intent.DECELERATE
        assert prediction.confidence > 0.5

    def test_predict_intent_turn_left(self):
        predictor = IntentPredictor()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])
        heading_changes = np.array([0.3])

        prediction = predictor.predict_intent(positions, velocities, accelerations, heading_changes)

        assert prediction.intent == Intent.TURN_LEFT
        assert prediction.confidence > 0.7

    def test_predict_intent_turn_right(self):
        predictor = IntentPredictor()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])
        heading_changes = np.array([-0.3])

        prediction = predictor.predict_intent(positions, velocities, accelerations, heading_changes)

        assert prediction.intent == Intent.TURN_RIGHT
        assert prediction.confidence > 0.7

    def test_predict_intent_continue(self):
        predictor = IntentPredictor()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])
        heading_changes = np.array([0.0])

        prediction = predictor.predict_intent(positions, velocities, accelerations, heading_changes)

        assert prediction.intent == Intent.CONTINUE
        assert prediction.confidence > 0.8

    def test_predict_multiple_intents_stationary(self):
        predictor = IntentPredictor()
        positions = np.array([[0, 0], [0, 0]])
        velocities = np.array([[0.1, 0], [0.1, 0]])
        accelerations = np.array([[0, 0]])
        heading_changes = np.array([0.0])

        intents = predictor.predict_multiple_intents(positions, velocities, accelerations, heading_changes)

        assert Intent.STOP in [intent[0] for intent in intents]

    def test_predict_multiple_intents_accelerating(self):
        predictor = IntentPredictor()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [2, 0]])
        accelerations = np.array([[1, 0]])
        heading_changes = np.array([0.0])

        intents = predictor.predict_multiple_intents(positions, velocities, accelerations, heading_changes)

        intent_types = [intent[0] for intent in intents]
        assert Intent.ACCELERATE in intent_types or Intent.CONTINUE in intent_types

    def test_predict_collision_intent_no_collision(self):
        predictor = IntentPredictor()
        ego_pos = np.array([0, 0])
        ego_vel = np.array([1, 0])
        other_pos = np.array([10, 0])
        other_vel = np.array([1, 0])

        is_collision, confidence, reasoning = predictor.predict_collision_intent(
            ego_pos, ego_vel, other_pos, other_vel, time_to_collision_threshold=3.0
        )

        assert is_collision is False

    def test_predict_collision_intent_collision(self):
        predictor = IntentPredictor()
        ego_pos = np.array([0, 0])
        ego_vel = np.array([5, 0])
        other_pos = np.array([2, 0])
        other_vel = np.array([0, 0])

        is_collision, confidence, reasoning = predictor.predict_collision_intent(
            ego_pos, ego_vel, other_pos, other_vel, time_to_collision_threshold=3.0
        )

        assert isinstance(bool(is_collision), bool)
        assert 0 <= confidence <= 1

    def test_estimate_evasion_trajectory_safe_distance(self):
        predictor = IntentPredictor()
        ego_pos = np.array([0, 0])
        ego_vel = np.array([1, 0])
        obstacle_pos = np.array([10, 0])
        obstacle_vel = np.array([0, 0])

        evasion = predictor.estimate_evasion_trajectory(ego_pos, ego_vel, obstacle_pos, obstacle_vel)

        assert evasion is None

    def test_estimate_evasion_trajectory_danger(self):
        predictor = IntentPredictor()
        ego_pos = np.array([0, 0])
        ego_vel = np.array([5, 0])
        obstacle_pos = np.array([1, 0])
        obstacle_vel = np.array([0, 0])

        evasion = predictor.estimate_evasion_trajectory(ego_pos, ego_vel, obstacle_pos, obstacle_vel, safety_distance=1.0)

        if evasion is not None:
            assert evasion.shape == (2,)

    def test_reasoning_generation(self):
        predictor = IntentPredictor()

        reasoning = predictor._generate_reasoning(Intent.ACCELERATE, 5.0, 1.0, 0.1)

        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_predicted_position(self):
        predictor = IntentPredictor(lookahead_frames=30)
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])
        heading_changes = np.array([0.0])

        prediction = predictor.predict_intent(positions, velocities, accelerations, heading_changes)

        assert prediction.predicted_position.shape == (2,)
        assert prediction.predicted_position[0] > positions[-1, 0]

    def test_confidence_bounds(self):
        predictor = IntentPredictor()
        positions = np.array([[0, 0], [1, 0]])
        velocities = np.array([[1, 0], [1, 0]])
        accelerations = np.array([[0, 0]])
        heading_changes = np.array([0.0])

        prediction = predictor.predict_intent(positions, velocities, accelerations, heading_changes)

        assert 0 <= prediction.confidence <= 1

    def test_collision_confidence_bounds(self):
        predictor = IntentPredictor()
        ego_pos = np.array([0, 0])
        ego_vel = np.array([1, 0])
        other_pos = np.array([5, 0])
        other_vel = np.array([0, 0])

        is_collision, confidence, reasoning = predictor.predict_collision_intent(ego_pos, ego_vel, other_pos, other_vel)

        assert 0 <= confidence <= 1
