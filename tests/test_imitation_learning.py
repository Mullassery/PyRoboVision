import pytest
import numpy as np
from pyrobovision.learning.imitation import ImitationLearner, DemonstrationDataset, Transition


class TestTransition:
    def test_initialization(self):
        obs = np.array([1, 2, 3])
        action = np.array([0.5, 0.2])
        next_obs = np.array([1.1, 2.1, 3.1])

        transition = Transition(
            observation=obs,
            action=action,
            next_observation=next_obs,
            reward=1.0,
            done=False,
        )

        assert np.allclose(transition.observation, obs)
        assert transition.reward == 1.0


class TestDemonstrationDataset:
    def test_initialization(self):
        transitions = [
            Transition(
                observation=np.array([1, 2]),
                action=np.array([0.5]),
                next_observation=np.array([1.1, 2.1]),
                reward=1.0,
                done=False,
            ),
        ]

        dataset = DemonstrationDataset(
            transitions=transitions,
            episode_lengths=[1],
            episode_rewards=[1.0],
        )

        assert len(dataset) == 1

    def test_get_statistics(self):
        transitions = [
            Transition(
                observation=np.array([i]),
                action=np.array([0.5]),
                next_observation=np.array([i + 1]),
                reward=float(i),
                done=i == 4,
            )
            for i in range(5)
        ]

        dataset = DemonstrationDataset(
            transitions=transitions,
            episode_lengths=[5],
            episode_rewards=[10.0],
        )

        stats = dataset.get_statistics()

        assert stats["num_episodes"] == 1
        assert stats["num_transitions"] == 5

    def test_normalize_observations(self):
        transitions = [
            Transition(
                observation=np.array([1.0, 2.0]),
                action=np.array([0.5]),
                next_observation=np.array([1.1, 2.1]),
                reward=1.0,
                done=False,
            ),
            Transition(
                observation=np.array([3.0, 4.0]),
                action=np.array([0.3]),
                next_observation=np.array([3.1, 4.1]),
                reward=2.0,
                done=True,
            ),
        ]

        dataset = DemonstrationDataset(
            transitions=transitions,
            episode_lengths=[2],
            episode_rewards=[3.0],
        )

        normalized = dataset.normalize_observations()

        assert len(normalized) == len(dataset)

    def test_augment_with_noise(self):
        transitions = [
            Transition(
                observation=np.array([1.0, 2.0]),
                action=np.array([0.5]),
                next_observation=np.array([1.1, 2.1]),
                reward=1.0,
                done=False,
            ),
        ]

        dataset = DemonstrationDataset(
            transitions=transitions,
            episode_lengths=[1],
            episode_rewards=[1.0],
        )

        augmented = dataset.augment_with_noise(noise_scale=0.01)

        assert len(augmented) == len(dataset)

    def test_train_test_split(self):
        transitions = [
            Transition(
                observation=np.array([i]),
                action=np.array([0.5]),
                next_observation=np.array([i + 1]),
                reward=1.0,
                done=i % 5 == 4,
            )
            for i in range(20)
        ]

        dataset = DemonstrationDataset(
            transitions=transitions,
            episode_lengths=[5, 5, 5, 5],
            episode_rewards=[5.0, 5.0, 5.0, 5.0],
        )

        train_set, test_set = dataset.train_test_split(train_ratio=0.8)

        assert len(train_set) + len(test_set) == len(dataset)
        assert len(train_set) > len(test_set)


class TestImitationLearner:
    def test_initialization(self):
        learner = ImitationLearner(obs_dim=4, action_dim=2)

        assert learner.obs_dim == 4
        assert learner.action_dim == 2
        assert len(learner.buffer) == 0

    def test_record_transition(self):
        learner = ImitationLearner(obs_dim=4, action_dim=2)

        learner.record_transition(
            observation=np.array([1, 2, 3, 4]),
            action=np.array([0.5, 0.2]),
            next_observation=np.array([1.1, 2.1, 3.1, 4.1]),
            reward=1.0,
            done=False,
        )

        assert len(learner.buffer) == 1

    def test_record_episode(self):
        learner = ImitationLearner(obs_dim=4, action_dim=2)

        for i in range(5):
            learner.record_transition(
                observation=np.array([i, i, i, i], dtype=float),
                action=np.array([0.5, 0.2]),
                next_observation=np.array([i + 1, i + 1, i + 1, i + 1], dtype=float),
                reward=1.0,
                done=i == 4,
            )

        assert len(learner.buffer) == 5
        assert len(learner.episode_lengths) == 1
        assert learner.episode_lengths[0] == 5

    def test_get_dataset(self):
        learner = ImitationLearner(obs_dim=4, action_dim=2)

        for i in range(5):
            learner.record_transition(
                observation=np.array([i, i, i, i], dtype=float),
                action=np.array([0.5, 0.2]),
                next_observation=np.array([i + 1, i + 1, i + 1, i + 1], dtype=float),
                reward=1.0,
                done=i == 4,
            )

        dataset = learner.get_dataset()

        assert len(dataset) == 5
        assert len(dataset.episode_lengths) == 1

    def test_sample_batch(self):
        learner = ImitationLearner(obs_dim=4, action_dim=2)

        for i in range(10):
            learner.record_transition(
                observation=np.array([i, i, i, i], dtype=float),
                action=np.array([0.5, 0.2]),
                next_observation=np.array([i + 1, i + 1, i + 1, i + 1], dtype=float),
                reward=1.0,
                done=False,
            )

        observations, actions = learner.sample_batch(batch_size=5)

        assert observations.shape[0] == 5
        assert actions.shape[0] == 5

    def test_sample_trajectory(self):
        learner = ImitationLearner(obs_dim=4, action_dim=2)

        for i in range(10):
            learner.record_transition(
                observation=np.array([i, i, i, i], dtype=float),
                action=np.array([0.5, 0.2]),
                next_observation=np.array([i + 1, i + 1, i + 1, i + 1], dtype=float),
                reward=1.0,
                done=False,
            )

        trajectory = learner.sample_trajectory(length=5)

        assert len(trajectory) == 5

    def test_get_statistics(self):
        learner = ImitationLearner(obs_dim=4, action_dim=2)

        for i in range(10):
            learner.record_transition(
                observation=np.array([i, i, i, i], dtype=float),
                action=np.array([0.5, 0.2]),
                next_observation=np.array([i + 1, i + 1, i + 1, i + 1], dtype=float),
                reward=float(i),
                done=i % 5 == 4,
            )

        stats = learner.get_statistics()

        assert stats["num_episodes"] == 2
        assert stats["num_transitions"] == 10
