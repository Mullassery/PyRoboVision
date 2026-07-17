import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    reward: float
    done: bool
    info: Dict = None

    def __post_init__(self):
        if self.info is None:
            self.info = {}


@dataclass
class DemonstrationDataset:
    transitions: List[Transition]
    episode_lengths: List[int]
    episode_rewards: List[float]

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Transition:
        return self.transitions[idx]

    def get_episode(self, episode_idx: int) -> List[Transition]:
        if episode_idx >= len(self.episode_lengths):
            return []

        start_idx = sum(self.episode_lengths[:episode_idx])
        end_idx = start_idx + self.episode_lengths[episode_idx]

        return self.transitions[start_idx:end_idx]

    def get_statistics(self) -> Dict:
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)

        return {
            "num_episodes": len(self.episode_lengths),
            "num_transitions": len(self.transitions),
            "avg_episode_length": float(np.mean(lengths)),
            "max_episode_length": int(np.max(lengths)),
            "min_episode_length": int(np.min(lengths)),
            "avg_episode_reward": float(np.mean(rewards)),
            "max_episode_reward": float(np.max(rewards)),
            "min_episode_reward": float(np.min(rewards)),
            "total_reward": float(np.sum(rewards)),
        }

    def normalize_observations(self) -> "DemonstrationDataset":
        all_obs = np.array([t.observation for t in self.transitions])

        obs_mean = np.mean(all_obs, axis=0)
        obs_std = np.std(all_obs, axis=0) + 1e-8

        normalized_transitions = []
        for t in self.transitions:
            normalized_obs = (t.observation - obs_mean) / obs_std
            normalized_next_obs = (t.next_observation - obs_mean) / obs_std

            normalized_transitions.append(
                Transition(
                    observation=normalized_obs,
                    action=t.action,
                    next_observation=normalized_next_obs,
                    reward=t.reward,
                    done=t.done,
                    info=t.info,
                )
            )

        return DemonstrationDataset(
            transitions=normalized_transitions,
            episode_lengths=self.episode_lengths.copy(),
            episode_rewards=self.episode_rewards.copy(),
        )

    def augment_with_noise(self, noise_scale: float = 0.01) -> "DemonstrationDataset":
        augmented_transitions = []

        for t in self.transitions:
            obs_noise = np.random.randn(*t.observation.shape) * noise_scale
            action_noise = np.random.randn(*t.action.shape) * noise_scale

            augmented_transitions.append(
                Transition(
                    observation=t.observation + obs_noise,
                    action=t.action + action_noise,
                    next_observation=t.next_observation + obs_noise,
                    reward=t.reward,
                    done=t.done,
                    info=t.info,
                )
            )

        return DemonstrationDataset(
            transitions=augmented_transitions,
            episode_lengths=self.episode_lengths.copy(),
            episode_rewards=self.episode_rewards.copy(),
        )

    def train_test_split(self, train_ratio: float = 0.8) -> Tuple["DemonstrationDataset", "DemonstrationDataset"]:
        num_episodes = len(self.episode_lengths)
        num_train = int(num_episodes * train_ratio)

        train_transitions = []
        train_lengths = []
        train_rewards = []

        test_transitions = []
        test_lengths = []
        test_rewards = []

        idx = 0
        for ep_idx, (length, reward) in enumerate(zip(self.episode_lengths, self.episode_rewards)):
            episode_data = self.transitions[idx : idx + length]

            if ep_idx < num_train:
                train_transitions.extend(episode_data)
                train_lengths.append(length)
                train_rewards.append(reward)
            else:
                test_transitions.extend(episode_data)
                test_lengths.append(length)
                test_rewards.append(reward)

            idx += length

        train_dataset = DemonstrationDataset(
            transitions=train_transitions,
            episode_lengths=train_lengths,
            episode_rewards=train_rewards,
        )

        test_dataset = DemonstrationDataset(
            transitions=test_transitions,
            episode_lengths=test_lengths,
            episode_rewards=test_rewards,
        )

        return train_dataset, test_dataset


class ImitationLearner:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_buffer_size: int = 100000,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_buffer_size = max_buffer_size
        self.buffer = deque(maxlen=max_buffer_size)
        self.episode_buffer = []
        self.episode_rewards = []
        self.episode_lengths = []

    def record_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        reward: float,
        done: bool,
        info: Optional[Dict] = None,
    ) -> None:
        transition = Transition(
            observation=observation,
            action=action,
            next_observation=next_observation,
            reward=reward,
            done=done,
            info=info or {},
        )

        self.buffer.append(transition)
        self.episode_buffer.append(transition)

        if done:
            self.episode_lengths.append(len(self.episode_buffer))
            self.episode_rewards.append(sum(t.reward for t in self.episode_buffer))
            self.episode_buffer = []

    def get_dataset(self) -> DemonstrationDataset:
        return DemonstrationDataset(
            transitions=list(self.buffer),
            episode_lengths=self.episode_lengths.copy(),
            episode_rewards=self.episode_rewards.copy(),
        )

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)

        observations = np.array([self.buffer[i].observation for i in indices])
        actions = np.array([self.buffer[i].action for i in indices])

        return observations, actions

    def sample_trajectory(self, length: int) -> List[Transition]:
        if len(self.buffer) == 0:
            return []

        start_idx = np.random.randint(0, len(self.buffer) - length + 1)
        return list(self.buffer)[start_idx : start_idx + length]

    def clear(self) -> None:
        self.buffer.clear()
        self.episode_buffer.clear()
        self.episode_lengths.clear()
        self.episode_rewards.clear()

    def get_statistics(self) -> Dict:
        if not self.episode_rewards:
            return {
                "num_episodes": 0,
                "num_transitions": 0,
                "avg_reward": 0.0,
                "avg_episode_length": 0.0,
            }

        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)

        return {
            "num_episodes": len(self.episode_rewards),
            "num_transitions": len(self.buffer),
            "avg_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "avg_episode_length": float(np.mean(lengths)),
            "max_episode_length": int(np.max(lengths)),
            "min_episode_length": int(np.min(lengths)),
        }
