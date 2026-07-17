import numpy as np
from typing import Tuple


class UncertaintyEstimator:
    def __init__(self, velocity_sigma: float = 0.5, accel_sigma: float = 0.2):
        self.velocity_sigma = velocity_sigma
        self.accel_sigma = accel_sigma

    def estimate_position_uncertainty(self, history: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        if len(history) < 2:
            return np.zeros(2), np.eye(2)

        velocity = history[-1] - history[-2]
        velocity_std = np.linalg.norm(velocity) * self.velocity_sigma

        if len(history) < 3:
            accel = np.zeros(2)
            accel_std = 0.0
        else:
            accel = (history[-1] - 2 * history[-2] + history[-3]) / (dt ** 2)
            accel_std = np.linalg.norm(accel) * self.accel_sigma

        uncertainty = np.array([velocity_std, accel_std])

        cov_matrix = np.array([
            [velocity_std ** 2, 0],
            [0, accel_std ** 2]
        ])

        return uncertainty, cov_matrix

    def predict_with_uncertainty(self, position: np.ndarray, velocity: np.ndarray, accel: np.ndarray,
                                  horizon: int, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        uncertainties = []

        for t in range(1, horizon + 1):
            pred = position + velocity * t * dt + 0.5 * accel * (t * dt) ** 2
            predictions.append(pred)

            vel_uncertainty = np.linalg.norm(velocity) * self.velocity_sigma * t
            accel_uncertainty = np.linalg.norm(accel) * self.accel_sigma * (t ** 2)
            total_uncertainty = np.sqrt(vel_uncertainty ** 2 + accel_uncertainty ** 2)
            uncertainties.append(total_uncertainty)

        return np.array(predictions), np.array(uncertainties)

    def compute_confidence_ellipse(self, cov_matrix: np.ndarray, n_std: float = 2.0) -> Tuple[float, float, float]:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        max_idx = np.argmax(eigenvalues)
        min_idx = np.argmin(eigenvalues)

        major_axis = n_std * np.sqrt(eigenvalues[max_idx])
        minor_axis = n_std * np.sqrt(eigenvalues[min_idx])
        angle = np.arctan2(eigenvectors[1, max_idx], eigenvectors[0, max_idx])

        return float(major_axis), float(minor_axis), float(angle)

    def estimate_velocity_uncertainty(self, velocity_history: np.ndarray) -> float:
        if len(velocity_history) < 2:
            return 0.0

        velocity_changes = np.linalg.norm(np.diff(velocity_history, axis=0), axis=1)
        return float(np.std(velocity_changes))

    def estimate_trajectory_uncertainty(self, position_history: np.ndarray, dt: float = 1.0) -> np.ndarray:
        if len(position_history) < 2:
            return np.zeros(1)

        velocities = np.diff(position_history, axis=0) / dt
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        uncertainties = velocity_magnitudes * self.velocity_sigma

        return uncertainties
