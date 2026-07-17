import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class TrainingConfig:
    obs_dim: int
    action_dim: int
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    hidden_sizes: List[int] = None
    use_data_augmentation: bool = True
    use_safety_constraints: bool = True
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    checkpoint_interval: int = 10

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256]

    def to_dict(self) -> Dict:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "hidden_sizes": self.hidden_sizes,
            "use_data_augmentation": self.use_data_augmentation,
            "use_safety_constraints": self.use_safety_constraints,
            "entropy_coefficient": self.entropy_coefficient,
            "value_loss_coefficient": self.value_loss_coefficient,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.training_log = []
        self.validation_log = []
        self.best_loss = float("inf")
        self.best_epoch = 0

    def train_epoch(
        self,
        model,
        train_observations: np.ndarray,
        train_actions: np.ndarray,
        epoch: int,
    ) -> Dict:
        num_batches = len(train_observations) // self.config.batch_size

        epoch_loss = 0.0
        epoch_mse = 0.0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = start_idx + self.config.batch_size

            batch_obs = train_observations[start_idx:end_idx]
            batch_actions = train_actions[start_idx:end_idx]

            loss_obj = model.train_step(batch_obs, batch_actions)

            epoch_loss += loss_obj.total_loss
            epoch_mse += loss_obj.mse_loss

        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_mse": avg_mse,
            "learning_rate": self.config.learning_rate,
        }

        self.training_log.append(log_entry)

        return log_entry

    def validate(
        self,
        model,
        val_observations: np.ndarray,
        val_actions: np.ndarray,
        epoch: int,
    ) -> Dict:
        mse_loss, kl_div = model.evaluate(val_observations, val_actions)

        total_loss = mse_loss + 0.1 * kl_div

        log_entry = {
            "epoch": epoch,
            "val_loss": total_loss,
            "val_mse": mse_loss,
            "val_kl": kl_div,
        }

        self.validation_log.append(log_entry)

        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.best_epoch = epoch

        return log_entry

    def train(
        self,
        model,
        train_observations: np.ndarray,
        train_actions: np.ndarray,
        val_observations: Optional[np.ndarray] = None,
        val_actions: Optional[np.ndarray] = None,
        early_stopping_patience: int = 10,
    ) -> Dict:
        patience_counter = 0
        best_val_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            train_log = self.train_epoch(model, train_observations, train_actions, epoch)

            if val_observations is not None and val_actions is not None:
                val_log = self.validate(model, val_observations, val_actions, epoch)

                if val_log["val_loss"] < best_val_loss:
                    best_val_loss = val_log["val_loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % self.config.checkpoint_interval == 0:
                print(f"Epoch {epoch + 1}: {train_log}")

        return {
            "num_epochs_trained": epoch + 1,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "training_history": self.training_log,
            "validation_history": self.validation_log,
        }

    def get_training_summary(self) -> Dict:
        if not self.training_log:
            return {}

        train_losses = np.array([h["train_loss"] for h in self.training_log])
        train_mses = np.array([h["train_mse"] for h in self.training_log])

        summary = {
            "total_epochs": len(self.training_log),
            "best_epoch": self.best_epoch,
            "best_loss": float(self.best_loss),
            "final_loss": float(self.training_log[-1]["train_loss"]),
            "avg_loss": float(np.mean(train_losses)),
            "min_loss": float(np.min(train_losses)),
            "max_loss": float(np.max(train_losses)),
            "avg_mse": float(np.mean(train_mses)),
            "loss_improvement": float(train_losses[0] - train_losses[-1]),
        }

        if self.validation_log:
            val_losses = np.array([h["val_loss"] for h in self.validation_log])
            summary["val_best_loss"] = float(np.min(val_losses))
            summary["val_final_loss"] = float(self.validation_log[-1]["val_loss"])

        return summary

    def plot_training_curve(self) -> Dict:
        if not self.training_log:
            return {}

        epochs = np.array([h["epoch"] for h in self.training_log])
        train_losses = np.array([h["train_loss"] for h in self.training_log])

        curve_data = {
            "epochs": epochs.tolist(),
            "train_losses": train_losses.tolist(),
        }

        if self.validation_log:
            val_epochs = np.array([h["epoch"] for h in self.validation_log])
            val_losses = np.array([h["val_loss"] for h in self.validation_log])

            curve_data["val_epochs"] = val_epochs.tolist()
            curve_data["val_losses"] = val_losses.tolist()

        return curve_data
