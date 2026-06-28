"""Image transforms for camera-frame batches.

Operate on a `[N, H, W, C]` array (the shape the loader yields per camera) and return a
transformed array. This is the CPU/NumPy implementation — the same op surface is what the
CV-CUDA (NVIDIA) and MLX backends will plug into later, so a transform script written today keeps
working as those land.

```python
from pyroboframes import transforms as T
tf = T.Compose([T.Resize(224, 224), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
```

Note: `Resize` uses nearest-neighbor sampling (dependency-free, deterministic); higher-quality
interpolation comes with the GPU backends.
"""

from __future__ import annotations

import numpy as np

#: Preference order for the transform compute backend (fastest/most-capable first).
TRANSFORM_BACKENDS = ("cvcuda", "mlx", "torch", "numpy")


def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def resolve_transform_backend(prefer: str = "auto") -> str:
    """Pick the transform compute backend via the fallback chain **CV-CUDA → MLX → Torch → NumPy**.

    ``prefer="auto"`` returns the first *available* backend in :data:`TRANSFORM_BACKENDS`; a
    specific ``prefer`` is honored if available, otherwise it degrades down the chain (never an
    error). NumPy is always available, so this always resolves. Native MLX/Torch/CV-CUDA ops
    plug in here; NumPy is the CPU fallback.
    """
    available = {
        "cvcuda": _importable("cvcuda"),
        "mlx": _importable("mlx.core"),
        "torch": _importable("torch"),
        "numpy": True,
    }
    if prefer != "auto":
        if prefer not in TRANSFORM_BACKENDS:
            raise ValueError(f"prefer must be one of {TRANSFORM_BACKENDS} or 'auto'")
        # Honor the preference if usable, else fall through the chain below it.
        chain = TRANSFORM_BACKENDS[TRANSFORM_BACKENDS.index(prefer):]
    else:
        chain = TRANSFORM_BACKENDS
    return next(b for b in chain if available[b])


class Compose:
    """Chain transforms, applied left to right."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"


class Resize:
    """Resize each frame to ``(height, width)``.

    ``interpolation`` is ``"bilinear"`` (default, half-pixel aligned) or ``"nearest"``. Bilinear
    returns ``float32``; nearest preserves the input dtype. Native MLX/Torch implementations
    dispatch via the backend (NumPy fallback if none available).
    """

    def __init__(self, height: int, width: int, interpolation: str = "bilinear"):
        if height <= 0 or width <= 0:
            raise ValueError("Resize dimensions must be positive")
        if interpolation not in ("bilinear", "nearest"):
            raise ValueError("interpolation must be 'bilinear' or 'nearest'")
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self._backend = None

    def __call__(self, x):
        if self._backend is None:
            self._backend = resolve_transform_backend()

        if self._backend == "cvcuda":
            return _resize_cvcuda(x, self.height, self.width, self.interpolation)
        elif self._backend == "mlx":
            return _resize_mlx(x, self.height, self.width, self.interpolation)
        elif self._backend == "torch":
            return _resize_torch(x, self.height, self.width, self.interpolation)
        elif self._backend == "numpy":
            _, h, w, _ = x.shape
            if self.interpolation == "nearest":
                ys = (np.arange(self.height) * h) // self.height
                xs = (np.arange(self.width) * w) // self.width
                return x[:, ys][:, :, xs]
            return _bilinear_resize(x, self.height, self.width)
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

    def __repr__(self) -> str:
        return f"Resize({self.height}, {self.width}, interpolation={self.interpolation!r})"


def _bilinear_resize(x, out_h: int, out_w: int):
    """Vectorized half-pixel-aligned bilinear resize of `[N, H, W, C]` -> `[N, out_h, out_w, C]`."""
    _, h, w, _ = x.shape
    xf = x.astype(np.float32)
    # Half-pixel centers map output coords back to input space, clamped to the edges.
    src_y = (np.arange(out_h, dtype=np.float32) + 0.5) * (h / out_h) - 0.5
    src_x = (np.arange(out_w, dtype=np.float32) + 0.5) * (w / out_w) - 0.5
    src_y = np.clip(src_y, 0, h - 1)
    src_x = np.clip(src_x, 0, w - 1)
    y0 = np.floor(src_y).astype(np.intp)
    x0 = np.floor(src_x).astype(np.intp)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (src_y - y0)[:, None]  # [out_h, 1]
    wx = (src_x - x0)[None, :]  # [1, out_w]

    # Gather the four neighbors: [N, out_h, out_w, C].
    top = xf[:, y0][:, :, x0] * (1 - wx)[..., None] + xf[:, y0][:, :, x1] * wx[..., None]
    bot = xf[:, y1][:, :, x0] * (1 - wx)[..., None] + xf[:, y1][:, :, x1] * wx[..., None]
    return (top * (1 - wy)[..., None] + bot * wy[..., None]).astype(np.float32)


class CenterCrop:
    """Crop the centered ``(height, width)`` region of each frame (clamped to the frame size)."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, x):
        _, h, w, _ = x.shape
        ch = min(self.height, h)
        cw = min(self.width, w)
        top = (h - ch) // 2
        left = (w - cw) // 2
        return x[:, top : top + ch, left : left + cw, :]

    def __repr__(self) -> str:
        return f"CenterCrop({self.height}, {self.width})"


class Normalize:
    """Scale to ``[0, 1]`` (divide by ``scale``) then standardize per channel: ``(x - mean) / std``.

    Returns ``float32``. ``mean``/``std`` are per-channel (length C). Native MLX/Torch
    implementations dispatch via the backend (NumPy fallback if none available).
    """

    def __init__(self, mean, std, scale: float = 255.0):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.scale = float(scale)
        self._backend = None

    def __call__(self, x):
        if self._backend is None:
            self._backend = resolve_transform_backend()

        if self._backend == "cvcuda":
            return _normalize_cvcuda(x, self.mean, self.std, self.scale)
        elif self._backend == "mlx":
            return _normalize_mlx(x, self.mean, self.std, self.scale)
        elif self._backend == "torch":
            return _normalize_torch(x, self.mean, self.std, self.scale)
        elif self._backend == "numpy":
            x = x.astype(np.float32) / self.scale
            return (x - self.mean) / self.std
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean.tolist()}, std={self.std.tolist()})"


class RandomHorizontalFlip:
    """Flip each frame left-right independently with probability ``p``. ``seed`` makes it
    reproducible (the RNG advances per call)."""

    def __init__(self, p: float = 0.5, seed: int | None = None):
        self.p = float(p)
        self._rng = np.random.default_rng(seed)

    def __call__(self, x):
        flip = self._rng.random(x.shape[0]) < self.p
        out = x.copy()
        out[flip] = x[flip, :, ::-1, :]
        return out

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomCrop:
    """Crop a random ``(height, width)`` window per frame (clamped to the frame size)."""

    def __init__(self, height: int, width: int, seed: int | None = None):
        self.height = height
        self.width = width
        self._rng = np.random.default_rng(seed)

    def __call__(self, x):
        n, h, w, c = x.shape
        ch, cw = min(self.height, h), min(self.width, w)
        tops = self._rng.integers(0, h - ch + 1, size=n)
        lefts = self._rng.integers(0, w - cw + 1, size=n)
        out = np.empty((n, ch, cw, c), dtype=x.dtype)
        for i in range(n):
            out[i] = x[i, tops[i] : tops[i] + ch, lefts[i] : lefts[i] + cw, :]
        return out

    def __repr__(self) -> str:
        return f"RandomCrop({self.height}, {self.width})"


# ───────────────────────────────────────────────────────────────────────────────
# MLX native implementations

def _resize_mlx(x, out_h: int, out_w: int, interpolation: str):
    """Resize using MLX (Apple Silicon GPU/CPU)."""
    import mlx.core as mx

    x_mx = mx.array(x)
    _, h, w, c = x_mx.shape

    if interpolation == "nearest":
        ys = (mx.arange(out_h) * h) // out_h
        xs = (mx.arange(out_w) * w) // out_w
        return mx.take(mx.take(x_mx, ys, axis=1), xs, axis=2)

    x_f = x_mx.astype(mx.float32)
    src_y = (mx.arange(out_h, dtype=mx.float32) + 0.5) * (h / out_h) - 0.5
    src_x = (mx.arange(out_w, dtype=mx.float32) + 0.5) * (w / out_w) - 0.5
    src_y = mx.clip(src_y, 0, h - 1)
    src_x = mx.clip(src_x, 0, w - 1)

    y0 = mx.floor(src_y).astype(mx.int32)
    x0 = mx.floor(src_x).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, h - 1)
    x1 = mx.minimum(x0 + 1, w - 1)

    wy = (src_y - y0.astype(mx.float32))[:, None]
    wx = (src_x - x0.astype(mx.float32))[None, :]

    top_l = mx.take(mx.take(x_f, y0, axis=1), x0, axis=2)
    top_r = mx.take(mx.take(x_f, y0, axis=1), x1, axis=2)
    bot_l = mx.take(mx.take(x_f, y1, axis=1), x0, axis=2)
    bot_r = mx.take(mx.take(x_f, y1, axis=1), x1, axis=2)

    top = top_l * (1 - wx)[..., None] + top_r * wx[..., None]
    bot = bot_l * (1 - wx)[..., None] + bot_r * wx[..., None]
    return (top * (1 - wy)[..., None] + bot * wy[..., None]).astype(mx.float32)


def _resize_torch(x, out_h: int, out_w: int, interpolation: str):
    """Resize using Torch (CPU/CUDA, includes MPS for macOS)."""
    import torch
    import torch.nn.functional as F

    x_t = torch.from_numpy(x).float()
    x_t = x_t.permute(0, 3, 1, 2)

    mode = "nearest" if interpolation == "nearest" else "bilinear"
    align_corners = (mode == "bilinear")
    x_resized = F.interpolate(x_t, size=(out_h, out_w), mode=mode, align_corners=align_corners)

    return x_resized.permute(0, 2, 3, 1).numpy()


def _normalize_mlx(x, mean, std, scale: float):
    """Normalize using MLX (Apple Silicon GPU/CPU)."""
    import mlx.core as mx

    x_mx = mx.array(x, dtype=mx.float32)
    x_mx = x_mx / scale
    mean_mx = mx.array(mean, dtype=mx.float32)
    std_mx = mx.array(std, dtype=mx.float32)
    return (x_mx - mean_mx) / std_mx


def _normalize_torch(x, mean, std, scale: float):
    """Normalize using Torch (CPU/CUDA, includes MPS for macOS)."""
    import torch

    x_t = torch.from_numpy(x).float()
    x_t = x_t / scale
    mean_t = torch.from_numpy(mean).float()
    std_t = torch.from_numpy(std).float()
    return ((x_t - mean_t) / std_t).numpy()


# ───────────────────────────────────────────────────────────────────────────────
# CV-CUDA native implementations (NVIDIA hardware only)

def _resize_cvcuda(x, out_h: int, out_w: int, interpolation: str):
    """Resize using CV-CUDA (NVIDIA GPU acceleration).

    Requires: pip install cvcuda-cu12 (or cvcuda-cu11 for CUDA 11.x)
    Interpolation: 'bilinear' uses linear filtering; 'nearest' uses nearest-neighbor.
    """
    try:
        import cvcuda
    except ImportError as e:
        raise ImportError(
            "CV-CUDA not installed. Install with: pip install cvcuda-cu12 "
            "(or cvcuda-cu11 for CUDA 11.x)"
        ) from e

    # [N, H, W, C] → [N, C, H, W] for cvcuda (typical tensor layout)
    n, h, w, c = x.shape
    x_reordered = np.transpose(x, (0, 3, 1, 2))  # [N, C, H, W]
    x_gpu = cvcuda.as_tensor(x_reordered)

    # Resize with specified interpolation
    interp = cvcuda.Interp.LINEAR if interpolation == "bilinear" else cvcuda.Interp.NEAREST
    x_resized = cvcuda.resize(
        x_gpu, (out_h, out_w), cvcuda.Interp.LINEAR if interpolation == "bilinear" else cvcuda.Interp.NEAREST
    )

    # Convert back to NumPy and reorder to [N, H, W, C]
    return np.transpose(x_resized.cuda().numpy(), (0, 2, 3, 1)).astype(np.float32)


def _normalize_cvcuda(x, mean, std, scale: float):
    """Normalize using CV-CUDA (NVIDIA GPU acceleration).

    Performs: (x / scale - mean) / std per channel.
    """
    try:
        import cvcuda
    except ImportError as e:
        raise ImportError(
            "CV-CUDA not installed. Install with: pip install cvcuda-cu12 "
            "(or cvcuda-cu11 for CUDA 11.x)"
        ) from e

    # [N, H, W, C] → [N, C, H, W] for cvcuda
    n, h, w, c = x.shape
    x_reordered = np.transpose(x, (0, 3, 1, 2))  # [N, C, H, W]
    x_gpu = cvcuda.as_tensor(x_reordered.astype(np.float32))

    # Scale, then subtract mean and divide by std (per channel)
    x_scaled = x_gpu / scale

    # Reshape mean/std for broadcasting: [C] → [1, C, 1, 1]
    mean_gpu = cvcuda.as_tensor(mean.astype(np.float32).reshape(1, c, 1, 1))
    std_gpu = cvcuda.as_tensor(std.astype(np.float32).reshape(1, c, 1, 1))

    x_norm = (x_scaled - mean_gpu) / std_gpu

    # Convert back to NumPy and reorder to [N, H, W, C]
    return np.transpose(x_norm.cuda().numpy(), (0, 2, 3, 1)).astype(np.float32)


class ColorJitter:
    """Multiply each frame by a random per-sample brightness factor in ``[1-b, 1+b]``.

    Integer inputs are clamped to ``[0, 255]`` and keep their dtype; float inputs pass through.
    """

    def __init__(self, brightness: float = 0.0, seed: int | None = None):
        if brightness < 0:
            raise ValueError("brightness must be >= 0")
        self.brightness = float(brightness)
        self._rng = np.random.default_rng(seed)

    def __call__(self, x):
        n = x.shape[0]
        factors = self._rng.uniform(
            1.0 - self.brightness, 1.0 + self.brightness, size=n
        ).astype(np.float32)
        out = x.astype(np.float32) * factors[:, None, None, None]
        if np.issubdtype(x.dtype, np.integer):
            return np.clip(out, 0, 255).astype(x.dtype)
        return out

    def __repr__(self) -> str:
        return f"ColorJitter(brightness={self.brightness})"
