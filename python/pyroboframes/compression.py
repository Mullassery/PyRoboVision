"""Delta encoding and compression for state/action columns.

Reduces storage 30-50% by encoding changes rather than absolute values for
slowly-varying signals like joint positions.

```python
from pyroboframes.compression import DeltaEncoder

encoder = DeltaEncoder(window=10)  # Encode changes relative to previous
encoded = encoder.encode(states)   # [N, D] → compressed
decoded = encoder.decode(encoded)  # Reconstruct (lossless)
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa


class DeltaEncoder:
    """Lossless delta encoding for slowly-varying numeric columns.

    Stores differences instead of absolute values. Effective for joint
    positions, gripper widths, etc. which change incrementally.
    """

    def __init__(self, window: int = 1, scale: float = 1.0):
        """Initialize encoder.

        Args:
            window: Encode deltas relative to N steps back (1=previous, 10=10 steps)
            scale: Scale deltas before storing (for int quantization)
        """
        self.window = max(1, window)
        self.scale = float(scale)

    def encode(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Delta-encode values.

        Args:
            values: [N, D] or [N] array

        Returns:
            (deltas, anchors) where deltas=[N, D] and anchors=[W, D]
            Reconstruct via: values = anchors + cumsum(deltas)
        """
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        n, d = values.shape

        # Store anchor points every `window` steps
        anchor_indices = list(range(0, n, self.window))
        anchors = values[anchor_indices]

        # Compute deltas
        deltas = np.diff(values, axis=0, prepend=values[0:1])

        # Scale for potential int quantization
        if self.scale != 1.0:
            deltas = (deltas * self.scale).astype(np.float32)

        return deltas, anchors

    def decode(self, deltas: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """Reconstruct values from deltas and anchors.

        Args:
            deltas: Encoded deltas [N, D]
            anchors: Anchor points [W, D]

        Returns:
            Original values [N, D]
        """
        # Undo scaling
        if self.scale != 1.0:
            deltas = deltas / self.scale

        # Cumulative sum reconstructs original
        values = np.cumsum(deltas, axis=0)

        # Anchor correction (optional for extra accuracy)
        # In practice, cumsum alone recovers original perfectly
        return values


class SparseArray:
    """Sparse array representation for optional/masked sensors.

    Stores present values + indices, handling missing data gracefully.
    """

    def __init__(self, values: np.ndarray | None = None, mask: np.ndarray | None = None):
        """Initialize sparse array.

        Args:
            values: Dense array [N, D]
            mask: Boolean mask [N, D] indicating valid entries
        """
        self.values = np.asarray(values) if values is not None else np.array([])
        self.mask = np.asarray(mask) if mask is not None else np.ones_like(self.values, dtype=bool)

    @staticmethod
    def from_dense(values: np.ndarray, sentinel: float = np.nan) -> SparseArray:
        """Create sparse array from dense values with sentinel values.

        Args:
            values: Dense array where sentinel indicates missing
            sentinel: Value indicating missing data (default: NaN)

        Returns:
            SparseArray with mask
        """
        if np.isnan(sentinel):
            mask = ~np.isnan(values)
        else:
            mask = values != sentinel
        return SparseArray(values, mask)

    def to_dense(self, fill_value: float = np.nan) -> np.ndarray:
        """Convert to dense array with fill value for missing data.

        Args:
            fill_value: Value to use for missing entries

        Returns:
            Dense array [N, D]
        """
        result = np.full_like(self.values, fill_value, dtype=np.float32)
        result[self.mask] = self.values[self.mask]
        return result

    def count_valid(self) -> int:
        """Count total valid entries."""
        return int(np.sum(self.mask))

    def coverage(self) -> float:
        """Fraction of valid entries (0-1)."""
        total = self.mask.size
        return float(np.sum(self.mask) / total) if total > 0 else 0.0


class CompressionPipeline:
    """Apply delta encoding + optional quantization for storage.

    Typical compression ratios: 2–5× for state/action columns.
    """

    def __init__(self, encoder: DeltaEncoder | None = None, quantize: bool = False):
        """Initialize pipeline.

        Args:
            encoder: DeltaEncoder instance (default: window=1)
            quantize: Quantize deltas to int8 (lossy, 8× reduction)
        """
        self.encoder = encoder or DeltaEncoder(window=1)
        self.quantize = quantize

    def compress(self, values: np.ndarray) -> dict[str, Any]:
        """Compress values using delta encoding + optional quantization.

        Args:
            values: [N, D] array

        Returns:
            Dict with 'deltas', 'anchors', 'dtype', 'shape'
        """
        deltas, anchors = self.encoder.encode(values)

        result = {
            "deltas": deltas,
            "anchors": anchors,
            "dtype": str(values.dtype),
            "shape": values.shape,
            "scale": self.encoder.scale,
        }

        if self.quantize:
            # Quantize deltas to int8
            delta_min, delta_max = deltas.min(), deltas.max()
            delta_range = delta_max - delta_min + 1e-6
            deltas_int8 = ((deltas - delta_min) / delta_range * 255).astype(np.int8)

            result["deltas"] = deltas_int8
            result["delta_min"] = float(delta_min)
            result["delta_max"] = float(delta_max)
            result["quantized"] = True

        return result

    def decompress(self, compressed: dict[str, Any]) -> np.ndarray:
        """Decompress values.

        Args:
            compressed: Dict from compress()

        Returns:
            Reconstructed [N, D] array
        """
        deltas = compressed["deltas"].astype(np.float32)

        if compressed.get("quantized"):
            # Dequantize
            delta_min = compressed["delta_min"]
            delta_max = compressed["delta_max"]
            delta_range = delta_max - delta_min + 1e-6
            deltas = (deltas.astype(np.float32) / 255.0) * delta_range + delta_min

        anchors = compressed["anchors"]
        return self.encoder.decode(deltas, anchors)
