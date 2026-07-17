import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PolicyState:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray


class PolicyNetwork:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = None,
        use_value_head: bool = True,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes or [128, 128]
        self.use_value_head = use_value_head

        self.policy_weights = []
        self.policy_biases = []

        self.value_weights = []
        self.value_biases = []

        self._initialize_networks()

        self.action_log_std = np.zeros(action_dim)
        self.action_std = np.exp(self.action_log_std)

    def _initialize_networks(self) -> None:
        layer_sizes = [self.obs_dim] + self.hidden_sizes

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))

            self.policy_weights.append(w)
            self.policy_biases.append(b)

            if self.use_value_head:
                self.value_weights.append(w.copy())
                self.value_biases.append(b.copy())

        policy_output_w = np.random.randn(self.hidden_sizes[-1], self.action_dim) * 0.01
        policy_output_b = np.zeros((1, self.action_dim))

        self.policy_weights.append(policy_output_w)
        self.policy_biases.append(policy_output_b)

        if self.use_value_head:
            value_output_w = np.random.randn(self.hidden_sizes[-1], 1) * 0.01
            value_output_b = np.zeros((1, 1))

            self.value_weights.append(value_output_w)
            self.value_biases.append(value_output_b)

    def forward_policy(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = observations

        for i in range(len(self.policy_weights) - 1):
            z = x @ self.policy_weights[i] + self.policy_biases[i]
            x = np.maximum(0, z)

        mu = x @ self.policy_weights[-1] + self.policy_biases[-1]

        return mu, self.action_std

    def forward_value(self, observations: np.ndarray) -> np.ndarray:
        if not self.use_value_head:
            return np.zeros((observations.shape[0], 1))

        x = observations

        for i in range(len(self.value_weights) - 1):
            z = x @ self.value_weights[i] + self.value_biases[i]
            x = np.maximum(0, z)

        value = x @ self.value_weights[-1] + self.value_biases[-1]

        return value

    def sample_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        mu, std = self.forward_policy(observation)

        if deterministic:
            action = mu[0]
            log_prob = -0.5 * np.sum(np.log(std ** 2))
        else:
            action = mu[0] + np.random.randn(self.action_dim) * std
            log_prob = -0.5 * np.sum(np.log(std ** 2) + ((action - mu[0]) ** 2) / (std ** 2))

        return action, float(log_prob)

    def compute_log_probs(self, actions: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
        log_probs = -0.5 * np.sum(np.log(std ** 2) + ((actions - mu) ** 2) / (std ** 2), axis=1)
        return log_probs

    def update_std(self, new_std: np.ndarray) -> None:
        self.action_std = new_std
        self.action_log_std = np.log(new_std)


class PolicyOptimizer:
    def __init__(
        self,
        policy: PolicyNetwork,
        learning_rate: float = 0.0003,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.policy = policy
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.update_history = []

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        gamma: float = 0.99,
        lambda_: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards)
        advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value - values[t]
            advantage = delta + gamma * lambda_ * advantage

            advantages[t] = advantage

        returns = advantages + values

        return advantages, returns

    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        num_epochs: int = 3,
    ) -> Dict:
        advantages, returns = self.compute_gae(rewards, values.flatten())

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(num_epochs):
            mu, std = self.policy.forward_policy(observations)
            log_probs = self.policy.compute_log_probs(actions, mu, std)

            ratio = np.exp(log_probs)

            policy_loss = -np.mean(log_probs * advantages * masks[:, 0])

            entropy = -np.mean(log_probs)

            if self.policy.use_value_head:
                predicted_values = self.policy.forward_value(observations)
                value_loss = np.mean((predicted_values - returns.reshape(-1, 1)) ** 2)

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            else:
                loss = policy_loss - self.entropy_coef * entropy
                value_loss = 0.0

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy

        avg_policy_loss = total_policy_loss / num_epochs
        avg_value_loss = total_value_loss / num_epochs
        avg_entropy = total_entropy / num_epochs

        update_info = {
            "policy_loss": float(avg_policy_loss),
            "value_loss": float(avg_value_loss),
            "entropy": float(avg_entropy),
            "total_loss": float(avg_policy_loss + avg_value_loss),
            "learning_rate": self.learning_rate,
        }

        self.update_history.append(update_info)

        return update_info

    def get_statistics(self) -> Dict:
        if not self.update_history:
            return {}

        policy_losses = np.array([h["policy_loss"] for h in self.update_history])
        entropies = np.array([h["entropy"] for h in self.update_history])

        return {
            "num_updates": len(self.update_history),
            "avg_policy_loss": float(np.mean(policy_losses)),
            "avg_entropy": float(np.mean(entropies)),
            "learning_rate": self.learning_rate,
        }
