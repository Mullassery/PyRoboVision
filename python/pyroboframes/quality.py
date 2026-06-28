"""Episode quality scoring for data curation and curriculum learning.

Computes per-episode quality metrics (diversity, sharpness, state variance, action complexity)
to identify and filter low-quality demonstrations before training.

```python
from pyroboframes.quality import EpisodeScorer

scorer = EpisodeScorer()
scores = scorer.score_episodes(ds)  # {episode_idx: {"diversity": ..., "sharpness": ...}}

# Filter to high-quality episodes
high_quality = [ep for ep, s in scores.items() if s["quality_score"] > 0.7]
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .dataframe import RoboticsDataFrame


class EpisodeScorer:
    """Compute quality metrics for episodes to identify low-quality data.

    Metrics computed:
    - **diversity**: entropy of action distribution (higher = more varied)
    - **sharpness**: Laplacian edge detection on frames (lower noise = sharper)
    - **state_variance**: variance in joint positions (higher = more exploration)
    - **action_magnitude**: RMS of action values (detection of near-zero actions)
    - **motion_smoothness**: inverse of acceleration magnitude (lower = smoother)
    - **quality_score**: weighted combination of above (0-1, normalized)
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        frame_sample_rate: int = 5,
    ):
        """Initialize scorer.

        Args:
            weights: Weights for quality_score computation. Default: balanced weights
            frame_sample_rate: Sample every Nth frame for sharpness (reduce compute)
        """
        self.weights = weights or {
            "diversity": 0.2,
            "sharpness": 0.2,
            "state_variance": 0.2,
            "action_magnitude": 0.2,
            "motion_smoothness": 0.2,
        }
        self.frame_sample_rate = frame_sample_rate

    def score_episodes(
        self,
        dataframe: RoboticsDataFrame,
        state_key: str = "observation.state",
        action_key: str = "action",
        camera_key: str | None = None,
    ) -> dict[int, dict[str, float]]:
        """Score all episodes in a RoboticsDataFrame.

        Args:
            dataframe: RoboticsDataFrame to score
            state_key: Column name for state (default: LeRobot convention)
            action_key: Column name for actions
            camera_key: Optional camera column for sharpness scoring

        Returns:
            Dict mapping episode_index → quality metrics dict
        """
        scores = {}
        for ep_idx in range(dataframe.num_episodes()):
            scores[ep_idx] = self.score_episode(
                dataframe,
                ep_idx,
                state_key=state_key,
                action_key=action_key,
                camera_key=camera_key,
            )
        return scores

    def score_episode(
        self,
        dataframe: RoboticsDataFrame,
        episode_index: int,
        state_key: str = "observation.state",
        action_key: str = "action",
        camera_key: str | None = None,
    ) -> dict[str, float]:
        """Score a single episode.

        Args:
            dataframe: RoboticsDataFrame
            episode_index: Episode to score
            state_key: State column name
            action_key: Action column name
            camera_key: Optional camera column for sharpness

        Returns:
            Dict with keys: diversity, sharpness, state_variance, action_magnitude,
            motion_smoothness, quality_score
        """
        scores: dict[str, float] = {}

        # Slice episode
        ep_slice = dataframe.slice(episode_index=episode_index)
        ep_table = ep_slice.to_pyarrow()

        # Action diversity (entropy of action distribution)
        if action_key in ep_table.column_names:
            actions = ep_table[action_key].combine_chunks().to_numpy()
            actions = np.array([a for a in actions if a is not None])
            if len(actions) > 0 and actions.ndim == 2:
                scores["diversity"] = float(self._action_diversity(actions))
            else:
                scores["diversity"] = 0.0
        else:
            scores["diversity"] = 0.0

        # State variance (exploration extent)
        if state_key in ep_table.column_names:
            states = ep_table[state_key].combine_chunks().to_numpy()
            states = np.array([s for s in states if s is not None])
            if len(states) > 0 and states.ndim == 2:
                scores["state_variance"] = float(self._state_variance(states))
            else:
                scores["state_variance"] = 0.0
        else:
            scores["state_variance"] = 0.0

        # Action magnitude (avoid trivial demonstrations)
        if action_key in ep_table.column_names:
            actions = ep_table[action_key].combine_chunks().to_numpy()
            actions = np.array([a for a in actions if a is not None])
            if len(actions) > 0 and actions.ndim == 2:
                scores["action_magnitude"] = float(self._action_magnitude(actions))
            else:
                scores["action_magnitude"] = 0.0
        else:
            scores["action_magnitude"] = 0.0

        # Motion smoothness (inverse of acceleration)
        if action_key in ep_table.column_names:
            actions = ep_table[action_key].combine_chunks().to_numpy()
            actions = np.array([a for a in actions if a is not None])
            if len(actions) > 1 and actions.ndim == 2:
                scores["motion_smoothness"] = float(self._motion_smoothness(actions))
            else:
                scores["motion_smoothness"] = 0.0
        else:
            scores["motion_smoothness"] = 0.0

        # Frame sharpness (if camera available)
        if camera_key and camera_key in ep_table.column_names:
            frames = ep_table[camera_key].combine_chunks().to_numpy()
            # Handle list/array of frames
            if len(frames) > 0 and isinstance(frames[0], (list, np.ndarray)):
                frames = np.array(frames)
                if frames.ndim == 4:  # [N, H, W, C]
                    scores["sharpness"] = float(
                        self._frame_sharpness(frames[:: self.frame_sample_rate])
                    )
                else:
                    scores["sharpness"] = 0.0
            else:
                scores["sharpness"] = 0.0
        else:
            scores["sharpness"] = 0.0

        # Weighted quality score
        scores["quality_score"] = self._compute_quality_score(scores)

        return scores

    @staticmethod
    def _action_diversity(actions: np.ndarray) -> float:
        """Entropy of action distribution (0-1, normalized).

        Actions with low entropy (repetitive) score low; varied actions score high.
        """
        if len(actions) == 0 or actions.ndim != 2:
            return 0.0

        # Discretize each action dimension, compute joint entropy
        n_bins = max(3, int(np.sqrt(len(actions))))
        entropies = []

        for dim in range(actions.shape[1]):
            col = actions[:, dim]
            if np.std(col) < 1e-6:
                entropies.append(0.0)
                continue

            hist, _ = np.histogram(col, bins=n_bins)
            hist = hist[hist > 0]
            if len(hist) == 0:
                entropies.append(0.0)
                continue

            p = hist / hist.sum()
            entropy = -np.sum(p * np.log(p + 1e-10))
            # Normalize by max entropy
            max_entropy = np.log(n_bins)
            entropies.append(min(1.0, entropy / max_entropy))

        return float(np.mean(entropies)) if entropies else 0.0

    @staticmethod
    def _state_variance(states: np.ndarray) -> float:
        """Variance across time (exploration extent, 0-1 normalized)."""
        if len(states) == 0 or states.ndim != 2:
            return 0.0

        # Per-dimension variance, normalized
        variances = []
        for dim in range(states.shape[1]):
            col = states[:, dim]
            if np.std(col) > 0:
                # Normalize by std; cap at 1.0
                v = np.var(col) / (np.std(col) ** 2 + 1e-6)
                variances.append(min(1.0, v))
            else:
                variances.append(0.0)

        return float(np.mean(variances)) if variances else 0.0

    @staticmethod
    def _action_magnitude(actions: np.ndarray) -> float:
        """RMS of action values (0-1, avoids near-zero actions)."""
        if len(actions) == 0 or actions.ndim != 2:
            return 0.0

        rms_per_dim = []
        for dim in range(actions.shape[1]):
            col = actions[:, dim]
            rms = np.sqrt(np.mean(col**2))
            # Normalize; assume typical action magnitude is 0-1
            rms_per_dim.append(min(1.0, rms))

        return float(np.mean(rms_per_dim)) if rms_per_dim else 0.0

    @staticmethod
    def _motion_smoothness(actions: np.ndarray) -> float:
        """Inverse of acceleration magnitude (smooth = high score).

        Smooth trajectories (low acceleration) indicate natural motion.
        """
        if len(actions) < 2 or actions.ndim != 2:
            return 0.0

        # Compute acceleration (second derivative)
        velocity = np.diff(actions, axis=0)
        if len(velocity) < 1:
            return 0.0

        acceleration = np.diff(velocity, axis=0)
        if len(acceleration) == 0:
            return 0.0

        # RMS acceleration
        accel_magnitude = np.sqrt(np.mean(acceleration**2))

        # Inverse: low acceleration → high smoothness
        smoothness = 1.0 / (1.0 + accel_magnitude)
        return float(smoothness)

    @staticmethod
    def _frame_sharpness(frames: np.ndarray) -> float:
        """Laplacian edge detection (sharp = high score).

        Blurry frames or heavy motion blur score low.
        """
        if len(frames) == 0 or frames.ndim != 4:
            return 0.0

        sharpnesses = []
        for frame in frames:
            # Convert to grayscale if needed
            if frame.shape[-1] == 3:
                gray = np.mean(frame, axis=2)
            else:
                gray = frame[..., 0]

            # Laplacian kernel for edge detection
            if gray.shape[0] < 3 or gray.shape[1] < 3:
                sharpnesses.append(0.0)
                continue

            laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
            edges = np.abs(
                np.convolve(gray.flatten(), laplacian.flatten(), mode="valid")
            )
            sharpness = np.sqrt(np.mean(edges**2))

            # Normalize (typical sharpness value ~10-50 for natural images)
            sharpness = min(1.0, sharpness / 50.0)
            sharpnesses.append(sharpness)

        return float(np.mean(sharpnesses)) if sharpnesses else 0.0

    def _compute_quality_score(self, metrics: dict[str, float]) -> float:
        """Compute weighted quality score (0-1)."""
        total_weight = sum(self.weights.values())
        score = 0.0

        for key, weight in self.weights.items():
            if key in metrics and key != "quality_score":
                score += metrics[key] * weight

        return float(score / total_weight) if total_weight > 0 else 0.0


def quality_percentile_filter(
    scores: dict[int, dict[str, float]], percentile: float = 50.0
) -> list[int]:
    """Filter episodes by quality_score percentile.

    Args:
        scores: Dict from EpisodeScorer.score_episodes()
        percentile: Keep episodes above this percentile (default: 50 = median)

    Returns:
        List of episode indices that meet the threshold
    """
    if not scores:
        return []

    quality_values = [s["quality_score"] for s in scores.values()]
    threshold = np.percentile(quality_values, percentile)
    return [ep_idx for ep_idx, s in scores.items() if s["quality_score"] >= threshold]
