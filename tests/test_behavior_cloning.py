import pytest
import numpy as np
from pyrobovision.learning.behavior_cloning import BehaviorCloningModel


class TestBehaviorCloningModel:
    def test_initialization(self):
        model = BehaviorCloningModel(obs_dim=10, action_dim=4)

        assert model.obs_dim == 10
        assert model.action_dim == 4
        assert len(model.weights) > 0

    def test_forward(self):
        model = BehaviorCloningModel(obs_dim=10, action_dim=4, hidden_sizes=[32, 32])

        observations = np.random.randn(8, 10)

        actions, cache = model.forward(observations)

        assert actions.shape == (8, 4)
        assert "activations" in cache

    def test_predict(self):
        model = BehaviorCloningModel(obs_dim=10, action_dim=4)

        observation = np.random.randn(10)

        action = model.predict(observation, deterministic=True)

        assert action.shape == (4,)

    def test_predict_stochastic(self):
        model = BehaviorCloningModel(obs_dim=10, action_dim=4)

        observation = np.random.randn(10)

        action1 = model.predict(observation, deterministic=False)
        action2 = model.predict(observation, deterministic=False)

        assert not np.allclose(action1, action2)

    def test_train_step(self):
        model = BehaviorCloningModel(obs_dim=10, action_dim=4, learning_rate=0.01)

        observations = np.random.randn(8, 10)
        actions = np.random.randn(8, 4)

        loss = model.train_step(observations, actions)

        assert loss.total_loss > 0
        assert len(model.training_history) == 1

    def test_evaluate(self):
        model = BehaviorCloningModel(obs_dim=10, action_dim=4)

        observations = np.random.randn(8, 10)
        actions = np.random.randn(8, 4)

        mse_loss, kl_div = model.evaluate(observations, actions)

        assert mse_loss >= 0
        assert kl_div >= 0

    def test_get_training_statistics(self):
        model = BehaviorCloningModel(obs_dim=10, action_dim=4, learning_rate=0.01)

        observations = np.random.randn(8, 10)
        actions = np.random.randn(8, 4)

        for _ in range(3):
            model.train_step(observations, actions)

        stats = model.get_training_statistics()

        assert stats["num_steps"] == 3
        assert "avg_loss" in stats

    def test_reset_history(self):
        model = BehaviorCloningModel(obs_dim=10, action_dim=4, learning_rate=0.01)

        observations = np.random.randn(8, 10)
        actions = np.random.randn(8, 4)

        model.train_step(observations, actions)

        assert len(model.training_history) == 1

        model.reset_history()

        assert len(model.training_history) == 0
