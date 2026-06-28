"""On-the-fly data augmentation for training robustness.

VLA models (ACT, Diffusion Policy) require augmented data. This module provides
common augmentations that work across frameworks (NumPy/MLX/Torch).

```python
from pyroboframes.augmentation import AugmentationPipeline

pipeline = AugmentationPipeline([
    RandomRotate(max_angle=15),
    RandomBrightness(max_delta=0.2),
    RandomNoise(std=0.01),
])

batch = loader.next()  # [N, H, W, 3]
augmented = pipeline(batch)  # Same shape, augmented
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


class Augmentation:
    """Base class for augmentations."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentation.

        Args:
            x: Input array [N, H, W, C] or [N, D]

        Returns:
            Augmented array (same shape)
        """
        raise NotImplementedError


class RandomRotate(Augmentation):
    """Random rotation for frame augmentation.

    Rotates in-plane (around optical axis) to simulate camera viewpoint variation.
    """

    def __init__(self, max_angle: float = 15.0):
        """Initialize.

        Args:
            max_angle: Maximum rotation in degrees
        """
        self.max_angle = float(max_angle)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Rotate frames.

        Args:
            x: [N, H, W, C] frames

        Returns:
            Rotated frames
        """
        if x.ndim != 4:
            return x

        n, h, w, c = x.shape
        angle = np.random.uniform(-self.max_angle, self.max_angle)

        # Rotate each frame
        result = np.zeros_like(x)
        for i in range(n):
            result[i] = self._rotate_frame(x[i], angle)
        return result

    @staticmethod
    def _rotate_frame(frame: np.ndarray, angle: float) -> np.ndarray:
        """Rotate single frame by angle (degrees)."""
        h, w = frame.shape[:2]
        center = (w / 2, h / 2)

        # Simple rotation via indexing (naive implementation)
        # Production would use cv2.warpAffine or similar
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        y, x = np.mgrid[:h, :w]
        x_c, y_c = x - center[0], y - center[1]
        x_rot = cos_a * x_c + sin_a * y_c + center[0]
        y_rot = -sin_a * x_c + cos_a * y_c + center[1]

        # Bilinear interpolation
        x_rot = np.clip(x_rot, 0, w - 1)
        y_rot = np.clip(y_rot, 0, h - 1)

        # Simplified: nearest neighbor
        x_idx = np.round(x_rot).astype(int)
        y_idx = np.round(y_rot).astype(int)
        return frame[y_idx, x_idx, :]


class RandomBrightness(Augmentation):
    """Random brightness/contrast adjustment."""

    def __init__(self, max_delta: float = 0.2):
        """Initialize.

        Args:
            max_delta: Max brightness change (fraction of range)
        """
        self.max_delta = float(max_delta)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Adjust brightness."""
        if x.ndim != 4:
            return x

        delta = np.random.uniform(-self.max_delta, self.max_delta)
        x_float = x.astype(np.float32) / 255.0
        x_bright = np.clip(x_float + delta, 0, 1)
        return (x_bright * 255).astype(x.dtype)


class RandomNoise(Augmentation):
    """Add Gaussian noise to frames."""

    def __init__(self, std: float = 0.01):
        """Initialize.

        Args:
            std: Standard deviation of noise (as fraction of range)
        """
        self.std = float(std)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Add noise."""
        if x.ndim != 4:
            return x

        x_float = x.astype(np.float32) / 255.0
        noise = np.random.normal(0, self.std, x_float.shape)
        x_noisy = np.clip(x_float + noise, 0, 1)
        return (x_noisy * 255).astype(x.dtype)


class RandomCrop(Augmentation):
    """Random crop (assumes padding or allow shrink)."""

    def __init__(self, crop_fraction: float = 0.1):
        """Initialize.

        Args:
            crop_fraction: Fraction to crop (0.1 = crop 10% of edge)
        """
        self.crop_fraction = float(crop_fraction)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Random crop."""
        if x.ndim != 4:
            return x

        n, h, w, c = x.shape
        crop_h = int(h * self.crop_fraction)
        crop_w = int(w * self.crop_fraction)

        result = np.zeros_like(x)
        for i in range(n):
            top = np.random.randint(0, crop_h + 1)
            left = np.random.randint(0, crop_w + 1)
            result[i] = x[i, top : top + h - crop_h, left : left + w - crop_w, :]
        return result


class RandomFlip(Augmentation):
    """Random horizontal/vertical flip."""

    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.0):
        """Initialize.

        Args:
            p_horizontal: Probability of horizontal flip
            p_vertical: Probability of vertical flip
        """
        self.p_horizontal = float(p_horizontal)
        self.p_vertical = float(p_vertical)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Flip frames."""
        if x.ndim != 4:
            return x

        result = x.copy()
        if np.random.random() < self.p_horizontal:
            result = result[:, :, ::-1, :]  # Flip horizontally
        if np.random.random() < self.p_vertical:
            result = result[:, ::-1, :, :]  # Flip vertically
        return result


class AugmentationPipeline:
    """Chain multiple augmentations."""

    def __init__(self, augmentations: list[Augmentation] | None = None):
        """Initialize.

        Args:
            augmentations: List of Augmentation instances
        """
        self.augmentations = augmentations or []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply all augmentations in sequence."""
        result = x.copy()
        for aug in self.augmentations:
            result = aug(result)
        return result

    def add(self, aug: Augmentation) -> AugmentationPipeline:
        """Add augmentation to pipeline.

        Args:
            aug: Augmentation instance

        Returns:
            Self (for chaining)
        """
        self.augmentations.append(aug)
        return self

    def __repr__(self) -> str:
        names = [type(a).__name__ for a in self.augmentations]
        return f"AugmentationPipeline([{', '.join(names)}])"
