import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class MotionModel(ABC):
    @abstractmethod
    def predict(self, positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
        pass


class ConstantVelocityModel(MotionModel):
    def predict(self, positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
        if len(positions) < 2:
            return positions[-1:].copy()

        velocity = positions[-1] - positions[-2]
        next_pos = positions[-1] + velocity
        return next_pos

    def predict_trajectory(self, positions: np.ndarray, horizon: int, dt: float = 1.0) -> np.ndarray:
        if len(positions) < 2:
            return np.tile(positions[-1], (horizon, 1))

        velocity = positions[-1] - positions[-2]
        trajectory = []
        for t in range(horizon):
            next_pos = positions[-1] + velocity * (t + 1)
            trajectory.append(next_pos)

        return np.array(trajectory)


class ConstantAccelerationModel(MotionModel):
    def predict(self, positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
        if len(positions) < 3:
            return ConstantVelocityModel().predict(positions, dt)

        velocity = positions[-1] - positions[-2]
        acceleration = (positions[-1] - 2 * positions[-2] + positions[-3]) / (dt ** 2)
        next_pos = positions[-1] + velocity * dt + 0.5 * acceleration * (dt ** 2)
        return next_pos

    def predict_trajectory(self, positions: np.ndarray, horizon: int, dt: float = 1.0) -> np.ndarray:
        if len(positions) < 3:
            return ConstantVelocityModel().predict_trajectory(positions, horizon, dt)

        velocity = positions[-1] - positions[-2]
        acceleration = (positions[-1] - 2 * positions[-2] + positions[-3]) / (dt ** 2)

        trajectory = []
        for t in range(1, horizon + 1):
            next_pos = positions[-1] + velocity * t * dt + 0.5 * acceleration * (t * dt) ** 2
            trajectory.append(next_pos)

        return np.array(trajectory)


_TORCH_REGRESSOR_CLS = None


def _get_regressor_cls():
    """Lazily define + cache the small GRU regressor `nn.Module` class.

    Deferred (rather than defined at module import time) so importing this
    module never requires torch — only actually instantiating
    `LearnedTrajectoryModel` does.
    """
    global _TORCH_REGRESSOR_CLS
    if _TORCH_REGRESSOR_CLS is None:
        import torch.nn as nn

        class _GRURegressor(nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.gru = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
                self.head = nn.Linear(hidden_size, 2)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1, :])

        _TORCH_REGRESSOR_CLS = _GRURegressor
    return _TORCH_REGRESSOR_CLS


class LearnedTrajectoryModel(MotionModel):
    """Small GRU-based learned motion model — an alternative to CV/CA for
    trajectories with curvature or motion patterns that don't match constant
    velocity/acceleration assumptions (e.g. turning, weaving).

    Operates on *velocity* (position-delta) sequences rather than absolute
    positions, so it generalizes across trajectories at different absolute
    locations/scales instead of memorizing specific coordinates.

    Unlike `ConstantVelocityModel`/`ConstantAccelerationModel`, this starts
    with randomly initialized weights and has nothing sensible to predict
    until trained. `predict()`/`predict_trajectory()` raise `RuntimeError`
    until `fit()` has been called at least once — an untrained network's
    output is meaningless noise, not a real prediction, and this class
    won't pretend otherwise.

    Requires `pip install torch` (not a pyrobovision hard dependency; see
    the `depth` optional-dependency group for how the rest of this project
    handles optional torch).
    """

    def __init__(self, history_len: int = 8, hidden_size: int = 32, device: str = "cpu"):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "LearnedTrajectoryModel requires PyTorch. Install it with "
                "`pip install torch`, or use ConstantVelocityModel/"
                "ConstantAccelerationModel for a dependency-free alternative."
            ) from exc

        self._torch = torch
        self.history_len = history_len
        self.device = device
        self._fitted = False
        self._net = _get_regressor_cls()(hidden_size).to(device)

    def fit(self, trajectories: List[np.ndarray], epochs: int = 300, lr: float = 1e-2) -> float:
        """Train on real (T, 2) position-sequence trajectories.

        Builds a supervised dataset of (window of `history_len` consecutive
        velocity vectors -> next velocity vector) across all trajectories,
        then runs real gradient descent (Adam + MSE) for `epochs` steps.
        Returns the final training loss.
        """
        torch = self._torch
        import torch.nn as nn

        X, Y = self._build_dataset(trajectories)
        if len(X) == 0:
            raise ValueError(
                "No training windows produced — each trajectory needs more "
                f"than history_len + 1 = {self.history_len + 1} positions."
            )

        X_t = torch.from_numpy(X).to(self.device)
        Y_t = torch.from_numpy(Y).to(self.device)

        opt = torch.optim.Adam(self._net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self._net.train()
        loss = None
        for _ in range(epochs):
            opt.zero_grad()
            pred = self._net(X_t)
            loss = loss_fn(pred, Y_t)
            loss.backward()
            opt.step()

        self._net.eval()
        self._fitted = True
        return float(loss.item())

    def _build_dataset(self, trajectories: List[np.ndarray]):
        X, Y = [], []
        for traj in trajectories:
            traj = np.asarray(traj, dtype=np.float32)
            if len(traj) < self.history_len + 2:
                continue
            velocity = traj[1:] - traj[:-1]
            for i in range(len(velocity) - self.history_len):
                X.append(velocity[i:i + self.history_len])
                Y.append(velocity[i + self.history_len])

        if not X:
            return (
                np.zeros((0, self.history_len, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "LearnedTrajectoryModel.predict() called before fit() — an "
                "untrained network's output is meaningless noise, not a "
                "real prediction. Call fit(trajectories) with example "
                "trajectory data first."
            )

    def _velocity_window(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        if len(positions) < 2:
            return np.zeros((self.history_len, 2), dtype=np.float32)

        velocity = positions[1:] - positions[:-1]
        if len(velocity) >= self.history_len:
            return velocity[-self.history_len:]

        pad = np.tile(velocity[0], (self.history_len - len(velocity), 1))
        return np.concatenate([pad, velocity], axis=0)

    def predict(self, positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
        self._require_fitted()
        torch = self._torch

        window = self._velocity_window(positions)
        with torch.no_grad():
            pred_velocity = (
                self._net(torch.from_numpy(window[None]).to(self.device))
                .cpu()
                .numpy()[0]
            )
        return np.asarray(positions[-1], dtype=np.float32) + pred_velocity * dt

    def predict_trajectory(self, positions: np.ndarray, horizon: int, dt: float = 1.0) -> np.ndarray:
        self._require_fitted()
        history = list(np.asarray(positions, dtype=np.float32))

        future = []
        for _ in range(horizon):
            next_pos = self.predict(np.array(history), dt=dt)
            future.append(next_pos)
            history.append(next_pos)

        return np.array(future)


class TrajectoryPredictor:
    def __init__(self, model: str = "cv", history_len: int = 10, **model_kwargs):
        self.history_len = history_len
        if model == "cv":
            self.model = ConstantVelocityModel()
        elif model == "ca":
            self.model = ConstantAccelerationModel()
        elif model == "learned":
            self.model = LearnedTrajectoryModel(**model_kwargs)
        else:
            raise ValueError(f"Unknown model {model!r}; expected 'cv', 'ca', or 'learned'")

    def update(self, position: np.ndarray, history: Optional[List[np.ndarray]] = None) -> None:
        if history is not None:
            self.history = history[-self.history_len:]
        else:
            self.history = [position]

    def predict_next(self, history: np.ndarray, dt: float = 1.0) -> np.ndarray:
        return self.model.predict(history, dt)

    def predict_trajectory(self, history: np.ndarray, horizon: int, dt: float = 1.0) -> np.ndarray:
        return self.model.predict_trajectory(history, horizon, dt)

    def compute_average_displacement_error(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        if len(predicted) != len(ground_truth):
            min_len = min(len(predicted), len(ground_truth))
            predicted = predicted[:min_len]
            ground_truth = ground_truth[:min_len]

        distances = np.linalg.norm(predicted - ground_truth, axis=1)
        return float(np.mean(distances))

    def compute_final_displacement_error(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        if len(predicted) > 0 and len(ground_truth) > 0:
            return float(np.linalg.norm(predicted[-1] - ground_truth[-1]))
        return 0.0
