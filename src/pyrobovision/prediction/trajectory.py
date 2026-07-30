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


class TrajectoryPredictor:
    def __init__(self, model: str = "cv", history_len: int = 10):
        self.history_len = history_len
        if model == "cv":
            self.model = ConstantVelocityModel()
        elif model == "ca":
            self.model = ConstantAccelerationModel()
        else:
            self.model = ConstantVelocityModel()

    def update(self, position: np.ndarray, history: Optional[List[np.ndarray]] = None) -> None:
        if history is not None:
            self.history = history[-self.history_len:]
        else:
            self.history = [position]

    def predict_next(self, history: np.ndarray, dt: float = 1.0) -> np.ndarray:
        return self.model.predict(history, dt)

    def predict_trajectory(self, history: np.ndarray, horizon: int, dt: float = 1.0) -> np.ndarray:
        if isinstance(self.model, ConstantAccelerationModel):
            return self.model.predict_trajectory(history, horizon, dt)
        else:
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
