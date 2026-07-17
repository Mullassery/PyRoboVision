import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from collections.abc import Callable


@dataclass
class BCLoss:
    mse_loss: float
    kl_divergence: float
    total_loss: float
    learning_rate: float

    def __str__(self) -> str:
        return f"MSE: {self.mse_loss:.4f}, KL: {self.kl_divergence:.4f}, Total: {self.total_loss:.4f}"


class BehaviorCloningModel:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = None,
        learning_rate: float = 0.001,
        use_dropout: bool = False,
        dropout_rate: float = 0.1,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes or [256, 256]
        self.learning_rate = learning_rate
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.weights = []
        self.biases = []
        self._initialize_network()

        self.training_history = []
        self.eval_history = []

    def _initialize_network(self) -> None:
        layer_sizes = [self.obs_dim] + self.hidden_sizes + [self.action_dim]

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(w)
            self.biases.append(b)

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, Dict]:
        activations = [observations]
        z_values = []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)

            a = np.maximum(0, z)

            if self.use_dropout:
                mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape) / (1 - self.dropout_rate)
                a = a * mask

            activations.append(a)

        z_output = activations[-1] @ self.weights[-1] + self.biases[-1]
        z_values.append(z_output)

        output = z_output
        activations.append(output)

        cache = {
            "activations": activations,
            "z_values": z_values,
        }

        return output, cache

    def backward(self, observations: np.ndarray, actions: np.ndarray, cache: Dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        batch_size = observations.shape[0]

        predictions = cache["activations"][-1]
        activations = cache["activations"]
        z_values = cache["z_values"]

        dL_doutput = (predictions - actions) / batch_size

        weight_gradients = []
        bias_gradients = []
        delta = dL_doutput

        for i in range(len(self.weights) - 1, -1, -1):
            dW = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)

            if i > 0:
                delta = (delta @ self.weights[i].T) * (z_values[i - 1] > 0)

        return weight_gradients, bias_gradients

    def train_step(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> BCLoss:
        predictions, cache = self.forward(observations)

        mse_loss = float(np.mean((predictions - actions) ** 2))
        kl_div = float(self._compute_kl_divergence(predictions, actions))
        total_loss = mse_loss + 0.1 * kl_div

        w_gradients, b_gradients = self.backward(observations, actions, cache)

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * w_gradients[i]
            self.biases[i] -= self.learning_rate * b_gradients[i]

        loss_obj = BCLoss(
            mse_loss=mse_loss,
            kl_divergence=kl_div,
            total_loss=total_loss,
            learning_rate=self.learning_rate,
        )

        self.training_history.append(loss_obj)

        return loss_obj

    def evaluate(self, observations: np.ndarray, actions: np.ndarray) -> Tuple[float, float]:
        predictions, _ = self.forward(observations)

        mse_loss = float(np.mean((predictions - actions) ** 2))
        kl_div = float(self._compute_kl_divergence(predictions, actions))

        return mse_loss, kl_div

    def predict(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        single_sample = False
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
            single_sample = True

        predictions, _ = self.forward(observations)

        if not deterministic:
            noise = np.random.randn(*predictions.shape) * 0.01
            predictions = predictions + noise

        if single_sample:
            return predictions[0]

        return predictions

    def _compute_kl_divergence(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        epsilon = 1e-8

        pred_probs = np.abs(predictions) + epsilon
        pred_probs /= np.sum(pred_probs, axis=1, keepdims=True)

        target_probs = np.abs(targets) + epsilon
        target_probs /= np.sum(target_probs, axis=1, keepdims=True)

        kl_div = np.sum(target_probs * (np.log(target_probs + epsilon) - np.log(pred_probs + epsilon)))

        return float(kl_div)

    def get_training_statistics(self) -> Dict:
        if not self.training_history:
            return {}

        losses = np.array([h.total_loss for h in self.training_history])
        mse_losses = np.array([h.mse_loss for h in self.training_history])

        return {
            "num_steps": len(self.training_history),
            "avg_loss": float(np.mean(losses)),
            "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)),
            "avg_mse": float(np.mean(mse_losses)),
            "loss_trend": "decreasing" if losses[-1] < losses[0] else "increasing",
        }

    def reset_history(self) -> None:
        self.training_history.clear()
        self.eval_history.clear()
