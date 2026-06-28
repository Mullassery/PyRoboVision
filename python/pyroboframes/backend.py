"""Runtime backend / device selection — the seam behind "Train Anywhere".

The same script targets CUDA on NVIDIA, MLX or MPS on Apple Silicon, or CPU, chosen at runtime
rather than by editing code. Resolution order for ``device="auto"``:

1. the ``PYROBOFRAMES_DEVICE`` environment variable, if set;
2. CUDA, if a CUDA-capable PyTorch is available;
3. on Apple Silicon: MLX if installed, else MPS (PyTorch) if available;
4. CPU (NumPy) otherwise.

**Zero-copy MLX path (pending mlx#2855):** On Apple Silicon with MLX, decoded video frames
are currently transferred via NumPy buffer protocol (low-copy via unified memory, not zero-copy).
Once MLX gains IOSurface/CVPixelBuffer direct initialization support, VideoToolbox-decoded
frames will pass directly to MLX arrays with zero copies — a ~3× speedup on the decode path.
"""

from __future__ import annotations

import os
import platform

VALID_DEVICES = ("cuda", "mps", "mlx", "cpu")


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:
        return None


def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401,PLC0415

        return True
    except Exception:
        return False


def resolve_device(device: str = "auto") -> str:
    """Resolve ``device`` to a concrete backend in :data:`VALID_DEVICES`.

    A concrete value (e.g. ``"cuda"``) is returned as-is; ``"auto"`` (or ``None``) is detected,
    honoring the ``PYROBOFRAMES_DEVICE`` override.
    """
    if not device:
        device = "auto"
    if device == "auto":
        device = os.environ.get("PYROBOFRAMES_DEVICE", "auto")
    if device != "auto":
        if device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {VALID_DEVICES} or 'auto' (got {device!r})")
        return device

    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if _is_apple_silicon():
        if _has_mlx():
            return "mlx"
        mps = getattr(getattr(torch, "backends", None), "mps", None) if torch else None
        if mps is not None and mps.is_available():
            return "mps"
    return "cpu"


def default_framework(device: str) -> str:
    """The native array framework for a (resolved) backend: ``"mlx"`` on Apple-MLX, ``"torch"`` on
    cuda/mps, ``"numpy"`` on cpu. This is the seam behind "no manual ``output=``": the loader can
    pick the right tensor type from the device alone."""
    return {"mlx": "mlx", "cuda": "torch", "mps": "torch", "cpu": "numpy"}[resolve_device(device)]


def to_backend(obj, device: str = "auto"):
    """Move a NumPy array — or a ``dict`` batch of them — to the framework native to ``device``.

    cpu → NumPy (unchanged); mlx → ``mlx.core.array``; cuda/mps → ``torch.Tensor`` on that device.
    The inverse of ``output=``: lets ``device="auto"`` choose the tensor type without code changes.
    """
    dev = resolve_device(device)
    if isinstance(obj, dict):
        return {k: to_backend(v, dev) for k, v in obj.items()}
    if dev == "cpu":
        return obj
    if dev == "mlx":
        import mlx.core as mx  # noqa: PLC0415

        return mx.array(obj)
    if dev in ("cuda", "mps"):
        import torch  # noqa: PLC0415

        return torch.as_tensor(obj).to(dev)
    raise ValueError(f"unsupported device {dev!r}")


def available_backends() -> dict[str, bool]:
    """Map each backend to whether it's usable in this environment (for diagnostics)."""
    torch = _torch()
    return {
        "cuda": bool(torch and torch.cuda.is_available()),
        "mps": bool(
            torch
            and getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_available()
        ),
        "mlx": _has_mlx() and _is_apple_silicon(),
        "cpu": True,
    }
