import pytest
import numpy as np
from pyrobovision.learning.imitation import ImitationLearner, DemonstrationDataset
from pyrobovision.learning.behavior_cloning import BehaviorCloningModel
from pyrobovision.learning.policy import PolicyNetwork, PolicyOptimizer
from pyrobovision.learning.safety import SafetyValidator
from pyrobovision.learning.training import TrainingConfig, Trainer


class TestV20Integration:
    def test_imitation_to_behavior_cloning(self):
        learner = ImitationLearner(obs_dim=8, action_dim=2)

        for i in range(50):
            obs = np.random.randn(8).astype(np.float32)
            action = np.random.randn(2).astype(np.float32) * 0.5
            next_obs = np.random.randn(8).astype(np.float32)

            learner.record_transition(obs, action, next_obs, 1.0, done=i % 10 == 9)

        dataset = learner.get_dataset()

        model = BehaviorCloningModel(obs_dim=8, action_dim=2)

        observations = np.array([t.observation for t in dataset.transitions])
        actions = np.array([t.action for t in dataset.transitions])

        for _ in range(5):
            model.train_step(observations, actions)

        assert len(model.training_history) == 5

    def test_full_learning_pipeline(self):
        learner = ImitationLearner(obs_dim=8, action_dim=2)

        for i in range(100):
            obs = np.random.randn(8).astype(np.float32)
            action = np.random.randn(2).astype(np.float32) * 0.5
            next_obs = np.random.randn(8).astype(np.float32)

            learner.record_transition(obs, action, next_obs, float(i % 10), done=i % 10 == 9)

        dataset = learner.get_dataset()

        train_set, test_set = dataset.train_test_split(train_ratio=0.8)

        model = BehaviorCloningModel(obs_dim=8, action_dim=2, learning_rate=0.01)

        config = TrainingConfig(obs_dim=8, action_dim=2, num_epochs=5, batch_size=16)
        trainer = Trainer(config)

        train_obs = np.array([t.observation for t in train_set.transitions])
        train_actions = np.array([t.action for t in train_set.transitions])

        test_obs = np.array([t.observation for t in test_set.transitions])
        test_actions = np.array([t.action for t in test_set.transitions])

        result = trainer.train(model, train_obs, train_actions, test_obs, test_actions)

        assert result["num_epochs_trained"] > 0

    def test_policy_learning_with_safety(self):
        policy = PolicyNetwork(obs_dim=8, action_dim=2, use_value_head=True)
        optimizer = PolicyOptimizer(policy, learning_rate=0.0003)
        validator = SafetyValidator(max_acceleration=5.0, max_steering=45.0)

        observations = np.random.randn(64, 8)
        actions = np.random.randn(64, 2)
        rewards = np.random.randn(64) * 10
        values = np.random.randn(64) * 2
        masks = np.ones((64, 1))

        update_info = optimizer.update(observations, actions, rewards, values, masks)

        assert "policy_loss" in update_info

        for obs, action in zip(observations[:5], actions[:5]):
            state = {"speed": abs(obs[0]) * 10, "min_distance_to_obstacle": abs(obs[1]) * 5}

            is_safe, violations = validator.validate_action(action, state)

            if not is_safe:
                corrected_action = validator.correct_action(action, state)
                assert corrected_action is not None

    def test_behavior_cloning_with_augmentation(self):
        learner = ImitationLearner(obs_dim=8, action_dim=2)

        for i in range(50):
            obs = np.random.randn(8).astype(np.float32)
            action = np.sin(obs[:2]).astype(np.float32)
            next_obs = np.random.randn(8).astype(np.float32)

            learner.record_transition(obs, action, next_obs, 1.0, done=i % 10 == 9)

        dataset = learner.get_dataset()

        normalized = dataset.normalize_observations()
        augmented = normalized.augment_with_noise(noise_scale=0.05)

        assert len(augmented) == len(dataset)

        model = BehaviorCloningModel(obs_dim=8, action_dim=2)

        train_obs = np.array([t.observation for t in augmented.transitions])
        train_actions = np.array([t.action for t in augmented.transitions])

        for _ in range(3):
            model.train_step(train_obs, train_actions)

    def test_multi_episode_learning(self):
        learner = ImitationLearner(obs_dim=8, action_dim=2)

        num_episodes = 10

        for ep in range(num_episodes):
            for step in range(20):
                obs = np.random.randn(8).astype(np.float32)
                action = np.random.randn(2).astype(np.float32) * 0.5
                next_obs = np.random.randn(8).astype(np.float32)

                learner.record_transition(
                    obs, action, next_obs, float(step), done=step == 19
                )

        stats = learner.get_statistics()

        assert stats["num_episodes"] == num_episodes
        assert stats["num_transitions"] == num_episodes * 20

    def test_safety_validation_throughout_learning(self):
        validator = SafetyValidator(
            max_acceleration=5.0, max_steering=45.0, min_distance=1.0
        )

        policy = PolicyNetwork(obs_dim=8, action_dim=2)

        safe_episodes = 0
        unsafe_episodes = 0

        for ep in range(10):
            episode_safe = True

            for step in range(10):
                obs = np.random.randn(8)
                action, _ = policy.sample_action(obs, deterministic=False)

                state = {
                    "speed": 15.0,
                    "min_distance_to_obstacle": 5.0,
                }

                is_safe, _ = validator.validate_action(action, state)

                if not is_safe:
                    episode_safe = False
                    action = validator.correct_action(action, state)

            if episode_safe:
                safe_episodes += 1
            else:
                unsafe_episodes += 1

        assert safe_episodes + unsafe_episodes == 10

    def test_end_to_end_training_scenario(self):
        config = TrainingConfig(
            obs_dim=8,
            action_dim=2,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.01,
        )

        learner = ImitationLearner(obs_dim=8, action_dim=2)

        for i in range(200):
            obs = np.random.randn(8).astype(np.float32)
            action = np.clip(np.random.randn(2) * 0.5, -1, 1).astype(np.float32)
            next_obs = np.random.randn(8).astype(np.float32)

            learner.record_transition(obs, action, next_obs, 1.0, done=i % 20 == 19)

        dataset = learner.get_dataset()
        train_set, test_set = dataset.train_test_split(train_ratio=0.8)

        model = BehaviorCloningModel(
            obs_dim=8,
            action_dim=2,
            learning_rate=config.learning_rate,
        )

        trainer = Trainer(config)

        train_obs = np.array([t.observation for t in train_set.transitions])
        train_actions = np.array([t.action for t in train_set.transitions])
        test_obs = np.array([t.observation for t in test_set.transitions])
        test_actions = np.array([t.action for t in test_set.transitions])

        result = trainer.train(
            model, train_obs, train_actions, test_obs, test_actions, early_stopping_patience=5
        )

        summary = trainer.get_training_summary()

        assert summary["total_epochs"] > 0
        assert summary["best_loss"] > 0

        validator = SafetyValidator()

        test_obs_sample = test_obs[:10]
        for obs in test_obs_sample:
            predicted_action = model.predict(obs, deterministic=True)

            state = {"speed": 15.0, "min_distance_to_obstacle": 5.0}

            is_safe, _ = validator.validate_action(predicted_action, state)

            if not is_safe:
                corrected = validator.correct_action(predicted_action, state)
                assert corrected is not None
