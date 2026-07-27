import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .patterns import MotionPattern, BehaviorPattern, MotionClassification, extract_motion_features


@dataclass
class BehaviorState:
    track_id: int
    current_pattern: MotionPattern
    current_behavior: BehaviorPattern
    confidence: float
    frame_count: int
    pattern_start_frame: int
    features: Dict
    history: List[Tuple[int, MotionPattern, float]]


class BehaviorAnalyzer:
    def __init__(
        self,
        velocity_threshold: float = 1.0,
        angular_velocity_threshold: float = 0.1,
        acceleration_threshold: float = 0.5,
        min_pattern_duration: int = 5,
    ):
        self.velocity_threshold = velocity_threshold
        self.angular_velocity_threshold = angular_velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.min_pattern_duration = min_pattern_duration
        self.track_states: Dict[int, BehaviorState] = {}

    def classify_motion_pattern(self, features: Dict) -> Tuple[MotionPattern, float]:
        velocity = features["velocity_magnitude"]
        angular_vel = features["angular_velocity"]
        acceleration = features["acceleration_magnitude"]

        if velocity < self.velocity_threshold:
            return MotionPattern.STOPPED, 0.95

        if angular_vel > self.angular_velocity_threshold:
            confidence = min(0.9, angular_vel / (self.angular_velocity_threshold * 2))
            return MotionPattern.TURNING, confidence

        if acceleration > self.acceleration_threshold:
            confidence = min(0.9, acceleration / (self.acceleration_threshold * 2))
            return MotionPattern.ACCELERATING, confidence

        if acceleration < -self.acceleration_threshold:
            confidence = min(0.9, abs(acceleration) / (self.acceleration_threshold * 2))
            return MotionPattern.DECELERATING, confidence

        confidence = 0.8
        return MotionPattern.MOVING_STRAIGHT, confidence

    def classify_behavior_pattern(self, motion_pattern: MotionPattern, feature_history: List[Dict]) -> BehaviorPattern:
        if motion_pattern == MotionPattern.STOPPED:
            return BehaviorPattern.STATIONARY

        if len(feature_history) == 0:
            return BehaviorPattern.UNKNOWN

        heading_changes = [f.get("heading_change", 0) for f in feature_history]
        if len(heading_changes) > 3 and np.mean(heading_changes[-3:]) > 0.3:
            return BehaviorPattern.LANE_CHANGE

        accelerations = [f.get("acceleration_magnitude", 0) for f in feature_history]
        if len(accelerations) > 3:
            avg_accel = np.mean(accelerations[-3:])
            if avg_accel > self.acceleration_threshold:
                return BehaviorPattern.ACCELERATION_EVENT
            if avg_accel < -self.acceleration_threshold:
                return BehaviorPattern.BRAKING_EVENT

        direction_variance = feature_history[-1].get("direction_variance", 0)
        if direction_variance > 0.5:
            return BehaviorPattern.CURVED_MOTION

        if motion_pattern == MotionPattern.TURNING:
            return BehaviorPattern.CURVED_MOTION

        return BehaviorPattern.LINEAR_MOTION

    def update(self, track_id: int, positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray, frame_id: int) -> BehaviorState:
        features = extract_motion_features(positions, velocities, accelerations)

        motion_pattern, confidence = self.classify_motion_pattern(features)

        if track_id not in self.track_states:
            self.track_states[track_id] = BehaviorState(
                track_id=track_id,
                current_pattern=motion_pattern,
                current_behavior=BehaviorPattern.UNKNOWN,
                confidence=confidence,
                frame_count=1,
                pattern_start_frame=frame_id,
                features=features,
                history=[],
            )
        else:
            state = self.track_states[track_id]
            state.frame_count += 1

            if motion_pattern != state.current_pattern:
                if state.frame_count >= self.min_pattern_duration:
                    state.history.append((state.pattern_start_frame, state.current_pattern, state.confidence))
                state.current_pattern = motion_pattern
                state.pattern_start_frame = frame_id
                state.confidence = confidence

            state.features = features

        state = self.track_states[track_id]
        state.current_behavior = self.classify_behavior_pattern(motion_pattern, [state.features])

        return state

    def get_state(self, track_id: int) -> Optional[BehaviorState]:
        return self.track_states.get(track_id)

    def get_all_states(self) -> Dict[int, BehaviorState]:
        return self.track_states.copy()

    def remove_track(self, track_id: int) -> None:
        if track_id in self.track_states:
            del self.track_states[track_id]

    def is_collision_course(self, track1_id: int, track2_id: int, distance_threshold: float = 2.0) -> bool:
        state1 = self.track_states.get(track1_id)
        state2 = self.track_states.get(track2_id)

        if state1 is None or state2 is None:
            return False

        vel1 = state1.features.get("velocity_magnitude", 0)
        vel2 = state2.features.get("velocity_magnitude", 0)

        if vel1 < 0.1 and vel2 < 0.1:
            return False

        return True

    def detect_lane_change(self, track_id: int, history_len: int = 10) -> bool:
        state = self.track_states.get(track_id)
        if state is None:
            return False

        heading_change = state.features.get("heading_change", 0)
        return heading_change > 0.3 and state.frame_count > history_len

    def get_motion_pattern_duration(self, track_id: int) -> int:
        state = self.track_states.get(track_id)
        if state is None:
            return 0
        return state.frame_count - state.pattern_start_frame

    def reset(self) -> None:
        self.track_states.clear()
