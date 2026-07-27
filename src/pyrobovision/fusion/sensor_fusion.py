import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class IMUData:
    timestamp: float
    accelerometer: np.ndarray  # [ax, ay, az] in m/s²
    gyroscope: np.ndarray      # [gx, gy, gz] in rad/s
    magnetometer: Optional[np.ndarray] = None  # [mx, my, mz] in Gauss


@dataclass
class GPSData:
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    speed: float
    heading: float
    accuracy: float  # Horizontal accuracy in meters


@dataclass
class FusionState:
    position: np.ndarray      # [x, y, z] in meters
    velocity: np.ndarray      # [vx, vy, vz] in m/s
    orientation: np.ndarray   # [roll, pitch, yaw] in radians
    covariance: np.ndarray    # 9x9 state covariance


class SensorFusionEngine:
    def __init__(
        self,
        origin_lat: float = 0.0,
        origin_lon: float = 0.0,
        origin_alt: float = 0.0,
        process_noise: float = 0.01,
        measurement_noise_imu: float = 0.1,
        measurement_noise_gps: float = 1.0,
    ):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.origin_alt = origin_alt

        self.process_noise = process_noise
        self.measurement_noise_imu = measurement_noise_imu
        self.measurement_noise_gps = measurement_noise_gps

        self.state = FusionState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.zeros(3),
            covariance=np.eye(9) * 0.1,
        )

        self.last_timestamp = None
        self.fusion_history = []

    def _gps_to_local(self, lat: float, lon: float, alt: float) -> np.ndarray:
        earth_radius = 6371000.0

        dlat = math.radians(lat - self.origin_lat)
        dlon = math.radians(lon - self.origin_lon)

        x = earth_radius * dlat
        y = earth_radius * dlon * math.cos(math.radians(self.origin_lat))
        z = alt - self.origin_alt

        return np.array([x, y, z])

    def _rotation_matrix_from_euler(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        R = np.array([
            [cy * cp, -sy * cr + cy * sp * sr, sy * sr + cy * sp * cr],
            [sy * cp, cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr],
            [-sp, cp * sr, cp * cr],
        ])

        return R

    def update_imu(self, imu_data: IMUData) -> FusionState:
        if self.last_timestamp is None:
            self.last_timestamp = imu_data.timestamp
            self.fusion_history.append({
                "timestamp": imu_data.timestamp,
                "type": "imu",
                "state": self.state,
            })
            return self.state

        dt = imu_data.timestamp - self.last_timestamp
        self.last_timestamp = imu_data.timestamp

        R = self._rotation_matrix_from_euler(*self.state.orientation)
        acc_world = R @ imu_data.accelerometer

        self.state.velocity += acc_world * dt
        self.state.position += self.state.velocity * dt

        self.state.orientation += imu_data.gyroscope * dt

        covariance_process = np.eye(9) * self.process_noise
        self.state.covariance += covariance_process

        self.fusion_history.append({
            "timestamp": imu_data.timestamp,
            "type": "imu",
            "state": self.state,
        })

        return self.state

    def update_gps(self, gps_data: GPSData) -> FusionState:
        gps_position = self._gps_to_local(gps_data.latitude, gps_data.longitude, gps_data.altitude)

        measurement_noise = self.measurement_noise_gps / (gps_data.accuracy + 1e-6)

        position_error = gps_position - self.state.position
        innovation_covariance = self.state.covariance[:3, :3] + measurement_noise * np.eye(3)

        kalman_gain = self.state.covariance[:3, :3] @ np.linalg.inv(innovation_covariance)

        self.state.position += kalman_gain @ position_error

        self.state.covariance[:3, :3] = (np.eye(3) - kalman_gain) @ self.state.covariance[:3, :3]

        if gps_data.speed > 0:
            heading_rad = math.radians(gps_data.heading)
            self.state.velocity[:2] = gps_data.speed * np.array([
                math.sin(heading_rad),
                math.cos(heading_rad),
            ])

        self.fusion_history.append({
            "timestamp": gps_data.timestamp,
            "type": "gps",
            "state": self.state,
        })

        return self.state

    def get_state_vector(self) -> np.ndarray:
        return np.concatenate([
            self.state.position,
            self.state.velocity,
            self.state.orientation,
        ])

    def get_covariance(self) -> np.ndarray:
        return self.state.covariance.copy()

    def reset(self) -> None:
        self.state = FusionState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.zeros(3),
            covariance=np.eye(9) * 0.1,
        )
        self.last_timestamp = None
        self.fusion_history.clear()

    def get_statistics(self) -> Dict:
        if not self.fusion_history:
            return {}

        imu_count = sum(1 for h in self.fusion_history if h["type"] == "imu")
        gps_count = sum(1 for h in self.fusion_history if h["type"] == "gps")

        return {
            "total_updates": len(self.fusion_history),
            "imu_updates": imu_count,
            "gps_updates": gps_count,
            "position": self.state.position.tolist(),
            "velocity": self.state.velocity.tolist(),
            "orientation": self.state.orientation.tolist(),
            "position_uncertainty": float(np.sqrt(np.trace(self.state.covariance[:3, :3]))),
        }
