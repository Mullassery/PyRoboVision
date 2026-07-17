import numpy as np
from typing import Tuple


class KalmanFilter:
    def __init__(self, dim_x: int = 4, dim_z: int = 2):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.F = np.eye(dim_x, dim_x)
        self.H = np.eye(dim_z, dim_x)
        self.R = np.eye(dim_z)
        self.Q = np.eye(dim_x)
        self.K = np.zeros((dim_x, dim_z))

    def predict(self, dt: float = 1.0) -> np.ndarray:
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.Q[0, 0] = dt ** 4 / 4
        self.Q[1, 1] = dt ** 4 / 4
        self.Q[2, 2] = dt ** 2
        self.Q[3, 3] = dt ** 2

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> None:
        z = z.reshape(-1, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + self.K @ y
        self.P = (np.eye(self.dim_x) - self.K @ self.H) @ self.P

    def initialize(self, z: np.ndarray, vx: float = 0.0, vy: float = 0.0) -> None:
        z = z.reshape(-1, 1)
        self.x[0:2] = z
        self.x[2:4, 0] = [vx, vy]

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.P.copy()

    def set_state(self, x: np.ndarray, P: np.ndarray) -> None:
        self.x = x.copy()
        self.P = P.copy()

    def mahalanobis_distance(self, z: np.ndarray) -> float:
        z = z.reshape(-1, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        dist_sq = float((y.T @ np.linalg.inv(S) @ y).item())
        return float(np.sqrt(dist_sq))
