"""Device-aware loader wrapper.

Wraps a (NumPy-output) loader from :meth:`RoboFrameDataset.loader`, applies image transforms to
camera-frame entries, and moves every array to the resolved backend — so the same loop runs on
MLX, MPS, CUDA, or CPU without code changes (see :mod:`pyroboframes.backend`).

```python
import pyroboframes as prf
from pyroboframes import transforms as T

ds = prf.RoboFrameDataset.from_path("…")
raw = ds.loader(batch_size=32, cameras=["observation.images.top"])  # output="numpy"
loader = prf.DataLoader(raw, transforms=T.Compose([T.Resize(224, 224)]), device="auto")
for batch in loader:        # arrays already transformed + on the chosen device/framework
    ...
```
"""

from __future__ import annotations

import time

from .backend import resolve_device


def _is_image(arr) -> bool:
    """Heuristic: a camera-frame batch is `[N, H, W, C]` with C in {1, 3, 4}."""
    return getattr(arr, "ndim", 0) == 4 and arr.shape[-1] in (1, 3, 4)


def _batch_size(batch: dict) -> int:
    """Rows in a batch (from `episode_index`, else the first array's leading dim)."""
    ep = batch.get("episode_index")
    if ep is not None:
        return int(ep.shape[0]) if hasattr(ep, "shape") else len(ep)
    for v in batch.values():
        if hasattr(v, "shape") and len(v.shape) > 0:
            return int(v.shape[0])
    return 0


def _to_device(batch: dict, device: str) -> dict:
    if device == "cpu":
        return batch  # leave as NumPy
    if device == "mlx":
        import mlx.core as mx  # noqa: PLC0415

        return {k: mx.array(v) for k, v in batch.items()}
    if device in ("cuda", "mps"):
        import torch  # noqa: PLC0415

        return {k: torch.as_tensor(v).to(device) for k, v in batch.items()}
    raise ValueError(f"unsupported device {device!r}")


class DataLoader:
    """Iterates ``inner``, applying ``transforms`` to image entries and moving to ``device``.

    ``inner`` should be a NumPy-output loader (the default). ``transforms`` is applied to every
    camera-frame array (``[N, H, W, C]``); other entries (state/action) pass through unchanged.
    """

    def __init__(self, inner, transforms=None, device: str = "cpu", on_batch=None):
        self._inner = inner
        self._transforms = transforms
        self.device = resolve_device(device)
        # `on_batch(index, batch, seconds)` is called after each batch is ready (profiling hook).
        self._on_batch = on_batch
        self._batches = 0
        self._frames = 0
        self._seconds = 0.0

    def __iter__(self):
        for i, batch in enumerate(self._inner):
            t0 = time.perf_counter()
            if self._transforms is not None:
                for key, value in list(batch.items()):
                    if _is_image(value):
                        batch[key] = self._transforms(value)
            batch = _to_device(batch, self.device)
            dt = time.perf_counter() - t0
            self._batches += 1
            self._frames += _batch_size(batch)
            self._seconds += dt
            if self._on_batch is not None:
                self._on_batch(i, batch, dt)
            yield batch

    @property
    def stats(self) -> dict:
        """Cumulative throughput so far: batches, frames, seconds, frames_per_s."""
        fps = self._frames / self._seconds if self._seconds > 0 else 0.0
        return {
            "batches": self._batches,
            "frames": self._frames,
            "seconds": self._seconds,
            "frames_per_s": fps,
        }

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def position(self) -> int:
        """Frames/rows consumed so far this epoch (delegated to the wrapped loader)."""
        return self._inner.position

    def __repr__(self) -> str:
        return f"DataLoader(device={self.device!r}, transforms={self._transforms!r})"
