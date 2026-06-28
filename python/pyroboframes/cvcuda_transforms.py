"""CV-CUDA transform operators for NVIDIA GPU acceleration.

Provides GPU-accelerated image transforms via CV-CUDA. Requires NVIDIA GPU + PyTorch +
cv-cuda package (pip install cv-cuda).

When available, these operators are preferred over CPU/MLX transforms for NVIDIA targets.
Stubs fallback to NumPy if cv-cuda is unavailable (e.g., non-NVIDIA systems).

```python
if has_cvcuda():
    resizer = CvCudaResize(224, 224)
    resizer(batch)  # Returns GPU tensor (cupy or torch)
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def has_cvcuda() -> bool:
    """Check if cv-cuda is available."""
    try:
        import cvcuda
        return True
    except ImportError:
        return False


class CvCudaResize:
    """Resize using CV-CUDA (NVIDIA GPU).

    Requires cv-cuda: pip install cv-cuda
    """

    def __init__(self, height: int, width: int, interpolation: str = "bilinear"):
        if height <= 0 or width <= 0:
            raise ValueError("Resize dimensions must be positive")
        if interpolation not in ("bilinear", "nearest"):
            raise ValueError("interpolation must be 'bilinear' or 'nearest'")

        self.height = height
        self.width = width
        self.interpolation = interpolation

        if not has_cvcuda():
            raise ImportError(
                "CvCudaResize requires cv-cuda (pip install cv-cuda)"
            )

        import cvcuda

        self.cvcuda = cvcuda
        self._interp_map = {
            "bilinear": cvcuda.Interp.LINEAR,
            "nearest": cvcuda.Interp.NEAREST,
        }

    def __call__(self, x: Any) -> Any:
        """Resize [N,H,W,C] tensor on GPU.

        Args:
            x: CUDA tensor (torch.Tensor with device='cuda')

        Returns:
            Resized tensor on GPU
        """
        import torch

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).cuda().float()
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected ndarray or torch.Tensor, got {type(x)}")

        if x.device.type != "cuda":
            x = x.cuda()

        # Convert torch [N,H,W,C] → [N,C,H,W] for CV-CUDA
        x_nchw = x.permute(0, 3, 1, 2)

        # Convert to CV-CUDA tensor
        x_cvcuda = self.cvcuda.as_tensor(x_nchw)

        # Resize
        interp = self._interp_map[self.interpolation]
        resized = self.cvcuda.resize(
            x_cvcuda,
            size=(self.height, self.width),
            interp=interp,
        )

        # Convert back to torch [N,C,H,W] → [N,H,W,C]
        result = torch.as_tensor(resized, device=x.device)
        return result.permute(0, 2, 3, 1)

    def __repr__(self) -> str:
        return f"CvCudaResize({self.height}, {self.width}, interpolation={self.interpolation!r})"


class CvCudaNormalize:
    """Normalize using CV-CUDA (NVIDIA GPU).

    Scales to [0, 1] then standardizes per-channel: (x - mean) / std.
    """

    def __init__(self, mean: list[float] | tuple[float, ...], std: list[float] | tuple[float, ...], scale: float = 255.0):
        import torch

        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 1, 1, -1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 1, 1, -1)
        self.scale = float(scale)

        if not has_cvcuda():
            raise ImportError(
                "CvCudaNormalize requires cv-cuda (pip install cv-cuda)"
            )

    def __call__(self, x: Any) -> Any:
        """Normalize [N,H,W,C] tensor on GPU.

        Args:
            x: CUDA tensor (torch.Tensor with device='cuda')

        Returns:
            Normalized tensor on GPU
        """
        import torch

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).cuda().float()
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected ndarray or torch.Tensor, got {type(x)}")

        if x.device.type != "cuda":
            x = x.cuda()

        mean = self.mean.to(x.device)
        std = self.std.to(x.device)

        return (x.float() / self.scale - mean) / std

    def __repr__(self) -> str:
        return f"CvCudaNormalize(mean=..., std=...)"


class CvCudaCenterCrop:
    """Center-crop using CV-CUDA (NVIDIA GPU)."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

        if not has_cvcuda():
            raise ImportError(
                "CvCudaCenterCrop requires cv-cuda (pip install cv-cuda)"
            )

    def __call__(self, x: Any) -> Any:
        """Crop [N,H,W,C] tensor on GPU.

        Args:
            x: CUDA tensor (torch.Tensor with device='cuda')

        Returns:
            Cropped tensor on GPU
        """
        import torch

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).cuda().float()
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected ndarray or torch.Tensor, got {type(x)}")

        _, h, w, _ = x.shape
        ch = min(self.height, h)
        cw = min(self.width, w)
        top = (h - ch) // 2
        left = (w - cw) // 2

        return x[:, top : top + ch, left : left + cw, :]

    def __repr__(self) -> str:
        return f"CvCudaCenterCrop({self.height}, {self.width})"
