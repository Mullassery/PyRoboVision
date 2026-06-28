"""TensorFlow/Keras support for PyRoboFrames dataloader.

Bridges PyRoboFrames to Keras training API.

```python
import tensorflow as tf
from pyroboframes.tensorflow_support import to_tf_dataset

ds = prf.RoboFrameDataset.from_path("…")
loader = ds.loader(batch_size=32, output="numpy")

# Convert to tf.data.Dataset
tf_ds = to_tf_dataset(loader, cache=True)

# Use in Keras
model.fit(tf_ds, epochs=10)
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ._core import Loader


def to_tf_dataset(
    loader: Loader,
    output_signature: dict[str, Any] | None = None,
    cache: bool = False,
    prefetch: int = 1,
) -> Any:
    """Convert PyRoboFrames loader to tf.data.Dataset.

    Args:
        loader: Loader from RoboFrameDataset
        output_signature: tf.TensorSpec dict (auto-inferred if None)
        cache: Cache dataset in memory (default: False)
        prefetch: Prefetch buffer size

    Returns:
        tf.data.Dataset yielding batches
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")

    if output_signature is None:
        # Infer from first batch
        first_batch = next(iter(loader))
        output_signature = _infer_tensor_spec(first_batch)

    # Create generator function
    def generator():
        for batch in loader:
            yield batch

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if cache:
        dataset = dataset.cache()

    if prefetch > 0:
        dataset = dataset.prefetch(prefetch)

    return dataset


def _infer_tensor_spec(batch: dict[str, Any]) -> dict[str, Any]:
    """Infer tf.TensorSpec from a batch dict.

    Args:
        batch: Dict of array batches

    Returns:
        Dict mapping keys to tf.TensorSpec
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow not installed")

    import numpy as np

    specs = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            specs[key] = tf.TensorSpec(shape=value.shape, dtype=tf.as_dtype(value.dtype))
        elif isinstance(value, (list, tuple)):
            # Assume uniform list of arrays
            if len(value) > 0 and isinstance(value[0], np.ndarray):
                arr = np.stack(value)
                specs[key] = tf.TensorSpec(shape=arr.shape, dtype=tf.as_dtype(arr.dtype))
        else:
            # Scalar or other
            specs[key] = tf.TensorSpec(shape=(), dtype=tf.as_dtype(type(value)))

    return specs


class KerasDataAdapter:
    """Keras data adapter for PyRoboFrames loaders.

    Allows passing loader directly to model.fit().
    """

    def __init__(self, loader: Loader, **kwargs: Any):
        """Initialize adapter.

        Args:
            loader: PyRoboFrames loader
            **kwargs: Additional arguments
        """
        self.loader = loader
        self.kwargs = kwargs

    def to_dataset(self) -> Any:
        """Convert to tf.data.Dataset for Keras."""
        return to_tf_dataset(
            self.loader,
            cache=self.kwargs.get("cache", False),
            prefetch=self.kwargs.get("prefetch", 1),
        )


def create_keras_model_for_robotics(
    state_dim: int,
    action_dim: int,
    sequence_length: int = 1,
    hidden_dims: list[int] | None = None,
) -> Any:
    """Create a simple Keras model for robotics (imitation learning).

    Args:
        state_dim: State vector dimensionality
        action_dim: Action vector dimensionality
        sequence_length: Temporal window size
        hidden_dims: Hidden layer dimensions (default: [256, 256])

    Returns:
        Compiled tf.keras.Model
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow not installed")

    if hidden_dims is None:
        hidden_dims = [256, 256]

    # Input: [batch, sequence_length, state_dim]
    state_input = tf.keras.Input(shape=(sequence_length, state_dim), name="state")

    # MLP: flatten sequence, predict action
    x = tf.keras.layers.Flatten()(state_input)
    for hidden_dim in hidden_dims:
        x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)

    # Output: action
    action_output = tf.keras.layers.Dense(action_dim, name="action")(x)

    model = tf.keras.Model(inputs=state_input, outputs=action_output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model
