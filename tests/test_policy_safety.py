import pytest
import numpy as np
from pyrobovision.learning.policy import PolicyNetwork, PolicyOptimizer
from pyrobovision.learning.safety import ConstraintChecker, SafetyValidator


class TestPolicyNetwork:
    def test_initialization(self):
        policy = PolicyNetwork(obs_dim=8, action_dim=2)

        assert policy.obs_dim == 8
        assert policy.action_dim == 2

    def test_forward_policy(self):
        policy = PolicyNetwork(obs_dim=8, action_dim=2)

        observations = np.random.randn(4, 8)

        mu, std = policy.forward_policy(observations)

        assert mu.shape == (4, 2)
        assert std.shape == (2,)

    def test_forward_value(self):
        policy = PolicyNetwork(obs_dim=8, action_dim=2, use_value_head=True)

        observations = np.random.randn(4, 8)

        values = policy.forward_value(observations)

        assert values.shape == (4, 1)

    def test_sample_action_deterministic(self):
        policy = PolicyNetwork(obs_dim=8, action_dim=2)

        observation = np.random.randn(8)

        action, log_prob = policy.sample_action(observation, deterministic=True)

        assert action.shape == (2,)
        assert isinstance(log_prob, float)

    def test_sample_action_stochastic(self):
        policy = PolicyNetwork(obs_dim=8, action_dim=2)

        observation = np.random.randn(8)

        action1, _ = policy.sample_action(observation, deterministic=False)
        action2, _ = policy.sample_action(observation, deterministic=False)

        assert not np.allclose(action1, action2)

    def test_compute_log_probs(self):
        policy = PolicyNetwork(obs_dim=8, action_dim=2)

        actions = np.random.randn(4, 2)
        mu = np.random.randn(4, 2)
        std = np.ones(2)

        log_probs = policy.compute_log_probs(actions, mu, std)

        assert log_probs.shape == (4,)


class TestPolicyOptimizer:
    def test_initialization(self):
        policy = PolicyNetwork(obs_dim=8, action_dim=2)
        optimizer = PolicyOptimizer(policy, learning_rate=0.0003)

        assert optimizer.learning_rate == 0.0003

    def test_compute_gae(self):
        policy = PolicyNetwork(obs_dim=8, action_dim=2)
        optimizer = PolicyOptimizer(policy)

        rewards = np.array([1.0, 1.0, 1.0, 0.0])
        values = np.array([1.0, 1.5, 1.0, 0.0])

        advantages, returns = optimizer.compute_gae(rewards, values)

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape

    def test_update(self):
        policy = PolicyNetwork(obs_dim=8, action_dim=2, use_value_head=True)
        optimizer = PolicyOptimizer(policy, learning_rate=0.0003)

        observations = np.random.randn(16, 8)
        actions = np.random.randn(16, 2)
        rewards = np.random.randn(16)
        values = np.random.randn(16)
        masks = np.ones((16, 1))

        update_info = optimizer.update(observations, actions, rewards, values, masks)

        assert "policy_loss" in update_info
        assert "entropy" in update_info

    def test_get_statistics(self):
        policy = PolicyNetwork(obs_dim=8, action_dim=2)
        optimizer = PolicyOptimizer(policy)

        observations = np.random.randn(16, 8)
        actions = np.random.randn(16, 2)
        rewards = np.random.randn(16)
        values = np.random.randn(16)
        masks = np.ones((16, 1))

        optimizer.update(observations, actions, rewards, values, masks)

        stats = optimizer.get_statistics()

        assert stats["num_updates"] == 1


class TestConstraintChecker:
    def test_initialization(self):
        checker = ConstraintChecker()

        assert len(checker.constraints) == 0

    def test_register_constraint(self):
        checker = ConstraintChecker()

        checker.register_constraint(
            "speed_limit",
            lambda state: state.get("speed", 0.0),
            threshold=30.0,
        )

        assert "speed_limit" in checker.constraints

    def test_check_constraint_satisfied(self):
        checker = ConstraintChecker()

        checker.register_constraint(
            "speed_limit",
            lambda state: state.get("speed", 0.0),
            threshold=30.0,
        )

        is_satisfied, value = checker.check_constraint("speed_limit", {"speed": 20.0})

        assert is_satisfied is True

    def test_check_constraint_violated(self):
        checker = ConstraintChecker()

        checker.register_constraint(
            "speed_limit",
            lambda state: state.get("speed", 0.0),
            threshold=30.0,
        )

        is_satisfied, value = checker.check_constraint("speed_limit", {"speed": 50.0})

        assert is_satisfied is False

    def test_check_all_constraints(self):
        checker = ConstraintChecker()

        checker.register_constraint("speed_limit", lambda s: s.get("speed", 0.0), threshold=30.0)
        checker.register_constraint("accel_limit", lambda s: abs(s.get("accel", 0.0)), threshold=5.0)

        state = {"speed": 20.0, "accel": 2.0}

        all_satisfied, violations = checker.check_all_constraints(state)

        assert all_satisfied is True


class TestSafetyValidator:
    def test_initialization(self):
        validator = SafetyValidator(max_acceleration=5.0, max_steering=45.0)

        assert validator.max_acceleration == 5.0
        assert validator.max_steering == 45.0

    def test_validate_action_safe(self):
        validator = SafetyValidator()

        action = np.array([2.0, 20.0])
        state = {"speed": 10.0, "min_distance_to_obstacle": 5.0}

        is_safe, violations = validator.validate_action(action, state)

        assert is_safe is True

    def test_validate_action_unsafe_acceleration(self):
        validator = SafetyValidator(max_acceleration=3.0)

        action = np.array([10.0, 20.0])
        state = {"speed": 10.0, "min_distance_to_obstacle": 5.0}

        is_safe, violations = validator.validate_action(action, state)

        assert is_safe is False

    def test_correct_action(self):
        validator = SafetyValidator(max_acceleration=3.0)

        action = np.array([10.0, 20.0])
        state = {"speed": 10.0, "min_distance_to_obstacle": 5.0}

        corrected = validator.correct_action(action, state)

        assert abs(corrected[0]) <= 3.0

    def test_get_safety_metrics(self):
        validator = SafetyValidator()

        action = np.array([2.0, 20.0])
        state = {"speed": 10.0, "min_distance_to_obstacle": 5.0}

        validator.validate_action(action, state)

        metrics = validator.get_safety_metrics()

        assert "total_violations" in metrics
