"""OKF Model Registry for PyRoboVision.

Computer vision model performance tracking, inference metadata,
and deployment history for robot vision systems.
"""

from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime


class OKFModelRegistry:
    """Track vision model performance and deployments."""

    def __init__(self, registry_dir: Path = None):
        self.registry_dir = registry_dir or Path.cwd() / "model_registry"
        self.registry_dir.mkdir(exist_ok=True)

    def register_model(self, model_name: str, model_path: str,
                      framework: str, accuracy: float) -> None:
        """Register a vision model."""
        entry = {
            'model_name': model_name,
            'path': model_path,
            'framework': framework,
            'accuracy': accuracy,
            'registered_at': datetime.now().isoformat()
        }

        log_file = self.registry_dir / "models.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_model_performance(self, model_name: str) -> Optional[Dict]:
        """Get model performance metrics."""
        log_file = self.registry_dir / "models.jsonl"
        if not log_file.exists():
            return None

        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                if entry['model_name'] == model_name:
                    return entry

        return None

    def get_best_model(self, framework: str = None) -> Optional[Dict]:
        """Get best performing model."""
        log_file = self.registry_dir / "models.jsonl"
        if not log_file.exists():
            return None

        best_model = None
        best_accuracy = -1

        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                if framework and entry['framework'] != framework:
                    continue

                if entry['accuracy'] > best_accuracy:
                    best_accuracy = entry['accuracy']
                    best_model = entry

        return best_model
