import pytest
import numpy as np
from pyrobovision.learning.training import TrainingConfig, Trainer
from pyrobovision.learning.behavior_cloning import BehaviorCloningModel


class TestTrainingConfig:
    def test_initialization(self):
        config = TrainingConfig(obs_dim=10, action_dim=4)

        assert config.obs_dim == 10
        assert config.action_dim == 4
        assert config.batch_size == 32

    def test_to_dict(self):
        config = TrainingConfig(obs_dim=10, action_dim=4, batch_size=64)

        config_dict = config.to_dict()

        assert config_dict["obs_dim"] == 10
        assert config_dict["batch_size"] == 64

    def test_save_and_load(self, tmp_path):
        config = TrainingConfig(obs_dim=10, action_dim=4, batch_size=64)

        config_path = tmp_path / "config.json"
        config.save(str(config_path))

        loaded_config = TrainingConfig.load(str(config_path))

        assert loaded_config.obs_dim == 10
        assert loaded_config.batch_size == 64


class TestTrainer:
    def test_initialization(self):
        config = TrainingConfig(obs_dim=10, action_dim=4)
        trainer = Trainer(config)

        assert trainer.config == config
        assert len(trainer.training_log) == 0

    def test_train_epoch(self):
        config = TrainingConfig(obs_dim=10, action_dim=4, batch_size=8)
        trainer = Trainer(config)

        model = BehaviorCloningModel(obs_dim=10, action_dim=4)

        observations = np.random.randn(32, 10)
        actions = np.random.randn(32, 4)

        log = trainer.train_epoch(model, observations, actions, epoch=0)

        assert "epoch" in log
        assert "train_loss" in log

    def test_validate(self):
        config = TrainingConfig(obs_dim=10, action_dim=4)
        trainer = Trainer(config)

        model = BehaviorCloningModel(obs_dim=10, action_dim=4)

        observations = np.random.randn(16, 10)
        actions = np.random.randn(16, 4)

        log = trainer.validate(model, observations, actions, epoch=0)

        assert "val_loss" in log
        assert "val_mse" in log

    def test_train(self):
        config = TrainingConfig(obs_dim=10, action_dim=4, num_epochs=3, batch_size=8)
        trainer = Trainer(config)

        model = BehaviorCloningModel(obs_dim=10, action_dim=4, learning_rate=0.01)

        train_obs = np.random.randn(32, 10)
        train_actions = np.random.randn(32, 4)

        val_obs = np.random.randn(16, 10)
        val_actions = np.random.randn(16, 4)

        result = trainer.train(
            model,
            train_obs,
            train_actions,
            val_obs,
            val_actions,
        )

        assert result["num_epochs_trained"] > 0
        assert "training_history" in result

    def test_get_training_summary(self):
        config = TrainingConfig(obs_dim=10, action_dim=4, num_epochs=3, batch_size=8)
        trainer = Trainer(config)

        model = BehaviorCloningModel(obs_dim=10, action_dim=4, learning_rate=0.01)

        train_obs = np.random.randn(32, 10)
        train_actions = np.random.randn(32, 4)

        trainer.train(model, train_obs, train_actions)

        summary = trainer.get_training_summary()

        assert "total_epochs" in summary
        assert "best_loss" in summary

    def test_plot_training_curve(self):
        config = TrainingConfig(obs_dim=10, action_dim=4, num_epochs=3, batch_size=8)
        trainer = Trainer(config)

        model = BehaviorCloningModel(obs_dim=10, action_dim=4, learning_rate=0.01)

        train_obs = np.random.randn(32, 10)
        train_actions = np.random.randn(32, 4)

        trainer.train(model, train_obs, train_actions)

        curve = trainer.plot_training_curve()

        assert "epochs" in curve
        assert "train_losses" in curve
