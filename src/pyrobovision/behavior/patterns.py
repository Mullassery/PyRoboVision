from enum import Enum
from dataclasses import dataclass
from typing import List
import numpy as np


class MotionPattern(Enum):
    STOPPED = "stopped"
    TURNING = "turning"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    MOVING_STRAIGHT = "moving_straight"
    COLLISION_COURSE = "collision_course"
    MERGING = "merging"
    DIVERGING = "diverging"
    UNKNOWN = "unknown"


class BehaviorPattern(Enum):
    STATIONARY = "stationary"
    LINEAR_MOTION = "linear_motion"
    CURVED_MOTION = "curved_motion"
    ERRATIC_MOTION = "erratic_motion"
    LANE_CHANGE = "lane_change"
    ACCELERATION_EVENT = "acceleration_event"
    BRAKING_EVENT = "braking_event"
    UNKNOWN = "unknown"


@dataclass
class MotionClassification:
    pattern: MotionPattern
    confidence: float
    duration: int
    start_frame: int
    velocity_magnitude: float
    angular_velocity: float


def extract_motion_features(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray) -> dict:
    if len(positions) == 0:
        return {
            "velocity_magnitude": 0.0,
            "angular_velocity": 0.0,
            "acceleration_magnitude": 0.0,
            "speed_variance": 0.0,
            "direction_variance": 0.0,
            "heading_change": 0.0,
        }

    velocity_magnitude = np.linalg.norm(velocities[-1]) if len(velocities) > 0 else 0.0
    velocity_magnitudes = np.array([np.linalg.norm(v) for v in velocities]) if len(velocities) > 0 else np.array([])

    if len(positions) > 1 and len(velocities) > 1:
        velocity_mags = np.array([np.linalg.norm(v) for v in velocities])
        if np.all(velocity_mags > 1e-6):
            directions = np.array([v / np.linalg.norm(v) for v in velocities])
            if len(directions) > 1:
                angular_velocity = np.mean(np.arccos(np.clip(np.dot(directions[:-1], directions[1:].T).diagonal(), -1, 1)))
            else:
                angular_velocity = 0.0
        else:
            angular_velocity = 0.0
    else:
        angular_velocity = 0.0

    acceleration_magnitude = np.linalg.norm(accelerations[-1]) if len(accelerations) > 0 else 0.0

    speed_variance = float(np.var(velocity_magnitudes)) if len(velocity_magnitudes) > 1 else 0.0
    direction_variance = float(np.var([np.arctan2(v[1], v[0]) for v in velocities])) if len(velocities) > 1 else 0.0

    heading_change = 0.0
    if len(positions) > 1:
        direction_start = np.arctan2(velocities[0, 1], velocities[0, 0])
        direction_end = np.arctan2(velocities[-1, 1], velocities[-1, 0])
        heading_change = abs(direction_end - direction_start)

    return {
        "velocity_magnitude": float(velocity_magnitude),
        "angular_velocity": float(angular_velocity),
        "acceleration_magnitude": float(acceleration_magnitude),
        "speed_variance": float(speed_variance),
        "direction_variance": float(direction_variance),
        "heading_change": float(heading_change),
    }
