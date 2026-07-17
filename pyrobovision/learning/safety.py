import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SafetyViolation:
    constraint_name: str
    severity: float
    message: str
    timestamp: float = 0.0

    def __str__(self) -> str:
        return f"[{self.constraint_name}] {self.message} (severity: {self.severity:.2f})"


class ConstraintChecker:
    def __init__(self):
        self.constraints: Dict[str, callable] = {}
        self.violation_history: List[SafetyViolation] = []
        self.constraint_stats: Dict[str, Dict] = {}

    def register_constraint(
        self,
        name: str,
        constraint_fn: callable,
        threshold: float = 0.5,
    ) -> None:
        self.constraints[name] = {
            "fn": constraint_fn,
            "threshold": threshold,
        }

        self.constraint_stats[name] = {
            "violations": 0,
            "avg_value": 0.0,
            "max_value": 0.0,
            "min_value": float("inf"),
        }

    def check_constraint(self, name: str, state: Dict) -> Tuple[bool, float]:
        if name not in self.constraints:
            return True, 0.0

        constraint_data = self.constraints[name]
        constraint_fn = constraint_data["fn"]
        threshold = constraint_data["threshold"]

        value = constraint_fn(state)

        is_satisfied = value <= threshold

        self.constraint_stats[name]["avg_value"] = (
            0.9 * self.constraint_stats[name]["avg_value"] + 0.1 * value
        )
        self.constraint_stats[name]["max_value"] = max(
            self.constraint_stats[name]["max_value"], value
        )
        self.constraint_stats[name]["min_value"] = min(
            self.constraint_stats[name]["min_value"], value
        )

        if not is_satisfied:
            self.constraint_stats[name]["violations"] += 1

        return is_satisfied, value

    def check_all_constraints(self, state: Dict) -> Tuple[bool, List[SafetyViolation]]:
        all_satisfied = True
        violations = []

        for constraint_name in self.constraints:
            is_satisfied, value = self.check_constraint(constraint_name, state)

            if not is_satisfied:
                all_satisfied = False

                threshold = self.constraints[constraint_name]["threshold"]
                severity = min(1.0, (value - threshold) / (threshold + 1e-8))

                violation = SafetyViolation(
                    constraint_name=constraint_name,
                    severity=severity,
                    message=f"Constraint '{constraint_name}' violated: {value:.4f} > {threshold:.4f}",
                )

                violations.append(violation)
                self.violation_history.append(violation)

        return all_satisfied, violations

    def get_statistics(self) -> Dict:
        stats = {}

        for name, data in self.constraint_stats.items():
            stats[name] = {
                "violations": data["violations"],
                "avg_value": float(data["avg_value"]),
                "max_value": float(data["max_value"]),
                "min_value": float(data["min_value"]),
            }

        return stats

    def clear_history(self) -> None:
        self.violation_history.clear()


class SafetyValidator:
    def __init__(self, max_acceleration: float = 5.0, max_steering: float = 45.0, min_distance: float = 1.0):
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.min_distance = min_distance

        self.constraint_checker = ConstraintChecker()
        self._register_default_constraints()

    def _register_default_constraints(self) -> None:
        self.constraint_checker.register_constraint(
            "acceleration_limit",
            lambda state: abs(state.get("acceleration", 0.0)),
            threshold=self.max_acceleration,
        )

        self.constraint_checker.register_constraint(
            "steering_limit",
            lambda state: abs(state.get("steering_angle", 0.0)),
            threshold=self.max_steering,
        )

        self.constraint_checker.register_constraint(
            "collision_risk",
            lambda state: 1.0 / (state.get("min_distance_to_obstacle", self.min_distance) + 1e-8),
            threshold=1.0 / self.min_distance,
        )

        self.constraint_checker.register_constraint(
            "speed_limit",
            lambda state: state.get("speed", 0.0),
            threshold=30.0,
        )

    def validate_action(self, action: np.ndarray, state: Dict) -> Tuple[bool, List[SafetyViolation]]:
        if len(action) >= 1:
            state["acceleration"] = action[0]

        if len(action) >= 2:
            state["steering_angle"] = action[1]

        is_safe, violations = self.constraint_checker.check_all_constraints(state)

        return is_safe, violations

    def validate_trajectory(self, trajectory: np.ndarray, states: List[Dict]) -> Tuple[bool, List[SafetyViolation]]:
        all_safe = True
        all_violations = []

        for action, state in zip(trajectory, states):
            is_safe, violations = self.validate_action(action, state)

            if not is_safe:
                all_safe = False
                all_violations.extend(violations)

        return all_safe, all_violations

    def correct_action(self, action: np.ndarray, state: Dict) -> np.ndarray:
        corrected_action = action.copy()

        is_safe, violations = self.validate_action(action, state)

        if not is_safe:
            for violation in violations:
                if "acceleration_limit" in violation.constraint_name:
                    corrected_action[0] = np.clip(corrected_action[0], -self.max_acceleration, self.max_acceleration)

                elif "steering_limit" in violation.constraint_name:
                    corrected_action[1] = np.clip(corrected_action[1], -self.max_steering, self.max_steering)

        return corrected_action

    def get_safety_metrics(self) -> Dict:
        stats = self.constraint_checker.get_statistics()

        total_violations = sum(s["violations"] for s in stats.values())

        return {
            "total_violations": total_violations,
            "constraint_statistics": stats,
            "violation_history_length": len(self.constraint_checker.violation_history),
        }
