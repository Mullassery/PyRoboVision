import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class Intent(Enum):
    CONTINUE = "continue"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"
    MERGE = "merge"
    DIVERGE = "diverge"
    EVASION = "evasion"
    UNKNOWN = "unknown"


@dataclass
class IntentPrediction:
    intent: Intent
    confidence: float
    lookahead_frames: int
    predicted_position: np.ndarray
    reasoning: str


class IntentPredictor:
    def __init__(self, lookahead_frames: int = 30, confidence_threshold: float = 0.6):
        self.lookahead_frames = lookahead_frames
        self.confidence_threshold = confidence_threshold

    def predict_intent(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        heading_changes: np.ndarray,
    ) -> IntentPrediction:
        if len(positions) < 2:
            return IntentPrediction(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                lookahead_frames=self.lookahead_frames,
                predicted_position=positions[-1] if len(positions) > 0 else np.array([0, 0]),
                reasoning="Insufficient history",
            )

        velocity = velocities[-1]
        acceleration = accelerations[-1] if len(accelerations) > 0 else np.array([0, 0])
        heading_change = heading_changes[-1] if len(heading_changes) > 0 else 0

        velocity_magnitude = np.linalg.norm(velocity)

        if velocity_magnitude < 0.5:
            return self._predict_stationary(positions[-1])

        vel_direction = velocity / (velocity_magnitude + 1e-6)
        accel_along_velocity = np.dot(acceleration, vel_direction)
        accel_magnitude = np.linalg.norm(acceleration)

        if accel_along_velocity > 0.5:
            intent, confidence = Intent.ACCELERATE, min(0.9, accel_along_velocity / 1.0)
        elif accel_along_velocity < -0.5:
            intent, confidence = Intent.DECELERATE, min(0.9, abs(accel_along_velocity) / 1.0)
        elif abs(heading_change) > 0.2:
            if heading_change > 0:
                intent, confidence = Intent.TURN_LEFT, 0.8
            else:
                intent, confidence = Intent.TURN_RIGHT, 0.8
        else:
            intent, confidence = Intent.CONTINUE, 0.85

        predicted_position = positions[-1] + velocity * self.lookahead_frames / 30.0

        return IntentPrediction(
            intent=intent,
            confidence=confidence,
            lookahead_frames=self.lookahead_frames,
            predicted_position=predicted_position,
            reasoning=self._generate_reasoning(intent, velocity_magnitude, accel_magnitude, heading_change),
        )

    def predict_multiple_intents(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        heading_changes: np.ndarray,
    ) -> List[Tuple[Intent, float]]:
        if len(positions) < 2:
            return [(Intent.UNKNOWN, 0.0)]

        velocity = velocities[-1]
        acceleration = accelerations[-1] if len(accelerations) > 0 else np.array([0, 0])
        heading_change = heading_changes[-1] if len(heading_changes) > 0 else 0

        velocity_magnitude = np.linalg.norm(velocity)
        accel_magnitude = np.linalg.norm(acceleration)

        intents = []

        if velocity_magnitude < 0.5:
            intents.append((Intent.STOP, 0.9))
        else:
            intents.append((Intent.CONTINUE, 0.7))

        if accel_magnitude > 0.5:
            intents.append((Intent.ACCELERATE, min(0.9, accel_magnitude / 1.0)))

        if accel_magnitude < -0.5:
            intents.append((Intent.DECELERATE, min(0.9, abs(accel_magnitude) / 1.0)))

        if heading_change > 0.2:
            intents.append((Intent.TURN_LEFT, 0.8))
        elif heading_change < -0.2:
            intents.append((Intent.TURN_RIGHT, 0.8))

        intents.sort(key=lambda x: x[1], reverse=True)

        return intents

    def _predict_stationary(self, position: np.ndarray) -> IntentPrediction:
        return IntentPrediction(
            intent=Intent.STOP,
            confidence=0.95,
            lookahead_frames=self.lookahead_frames,
            predicted_position=position.copy(),
            reasoning="Vehicle is stationary",
        )

    def _generate_reasoning(self, intent: Intent, velocity: float, acceleration: float, heading_change: float) -> str:
        parts = []

        if velocity < 0.5:
            parts.append("low velocity")
        elif velocity > 2.0:
            parts.append("high velocity")

        if acceleration > 0.5:
            parts.append("positive acceleration")
        elif acceleration < -0.5:
            parts.append("negative acceleration (braking)")

        if abs(heading_change) > 0.2:
            direction = "left" if heading_change > 0 else "right"
            parts.append(f"heading change to {direction}")

        reasoning = f"{intent.value}: " + ", ".join(parts) if parts else intent.value

        return reasoning

    def predict_collision_intent(
        self,
        ego_position: np.ndarray,
        ego_velocity: np.ndarray,
        other_position: np.ndarray,
        other_velocity: np.ndarray,
        time_to_collision_threshold: float = 3.0,
    ) -> Tuple[bool, float, str]:
        relative_position = other_position - ego_position
        relative_velocity = other_velocity - ego_velocity

        distance = np.linalg.norm(relative_position)
        relative_speed = np.linalg.norm(relative_velocity)

        if relative_speed < 0.1:
            return False, 1.0, "No relative motion"

        time_to_collision = distance / (relative_speed + 1e-6)

        is_collision = time_to_collision < time_to_collision_threshold

        confidence = 1.0 - (time_to_collision / time_to_collision_threshold) if time_to_collision > 0 else 0.0
        confidence = max(0.0, min(1.0, confidence))

        reasoning = f"TTC: {time_to_collision:.2f}s, distance: {distance:.2f}m"

        return is_collision, confidence, reasoning

    def estimate_evasion_trajectory(
        self,
        ego_position: np.ndarray,
        ego_velocity: np.ndarray,
        obstacle_position: np.ndarray,
        obstacle_velocity: np.ndarray,
        safety_distance: float = 1.0,
    ) -> Optional[np.ndarray]:
        relative_pos = obstacle_position - ego_position
        distance = np.linalg.norm(relative_pos)

        if distance > safety_distance * 3:
            return None

        perpendicular = np.array([-relative_pos[1], relative_pos[0]])
        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)

        evasion_direction = perpendicular * np.sign(np.dot(perpendicular, ego_velocity))

        current_speed = np.linalg.norm(ego_velocity)
        evasion_velocity = evasion_direction * current_speed

        return evasion_velocity
