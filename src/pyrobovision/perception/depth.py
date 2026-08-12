"""Monocular depth estimation.

Two backends are available via ``DepthEstimator(model=...)``:

- ``"heuristic"`` (default): a fast, dependency-free Sobel-edge-based depth
  *placeholder*. It does not run any neural network and does not produce
  metrically accurate depth — it assigns closer/farther values based on local
  edge density as a crude proxy. Useful for exercising the rest of the
  pipeline (3D bbox conversion, occupancy grids, etc.) without installing
  PyTorch. Do not use it for anything depth-accuracy sensitive.
- ``"midas"``: real monocular depth inference using Intel ISL's MiDaS
  (``MiDaS_small``) via ``torch.hub``. Produces genuine learned *relative*
  (inverse) depth, converted here into an ordered depth map. This requires
  ``pip install "pyrobovision[depth]"`` (installs PyTorch) and, on first use,
  a one-time download of the pretrained weights (~90MB) from GitHub, which
  requires network access. MiDaS output is relative depth (correct ordering
  of near/far, not metric meters) unless you calibrate it against a known
  reference distance.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter as _ndi_median_filter
from scipy.ndimage import sobel as _ndi_sobel

_VALID_MODELS = ("heuristic", "midas")


@dataclass
class DepthMap:
    data: np.ndarray
    scale: float
    shift: float
    min_depth: float
    max_depth: float

    def get_depth(self, x: int, y: int) -> float:
        if 0 <= y < self.data.shape[0] and 0 <= x < self.data.shape[1]:
            return float(self.data[y, x])
        return self.max_depth

    def get_depth_batch(self, points: np.ndarray) -> np.ndarray:
        """Vectorized lookup of depth values at pixel coordinates.

        ``points`` is an (N, 2) array of (x, y) pixel coordinates. Points
        outside the image bounds return ``max_depth``, matching
        ``get_depth``'s out-of-bounds behavior.
        """
        points = np.asarray(points)
        if points.size == 0:
            return np.zeros((0,), dtype=self.data.dtype)

        h, w = self.data.shape
        xs = points[:, 0].astype(np.int64)
        ys = points[:, 1].astype(np.int64)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

        depths = np.full(len(points), self.max_depth, dtype=np.float64)
        depths[valid] = self.data[ys[valid], xs[valid]]
        return depths

    def upsample(self, factor: int = 2) -> "DepthMap":
        """Nearest-neighbor upsample by an integer factor (vectorized)."""
        upsampled = np.kron(self.data, np.ones((factor, factor), dtype=self.data.dtype))

        return DepthMap(
            data=upsampled,
            scale=self.scale,
            shift=self.shift,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

    def downsample(self, factor: int = 2) -> "DepthMap":
        h, w = self.data.shape
        new_h, new_w = h // factor, w // factor
        # Reshape into (new_h, factor, new_w, factor) blocks and average —
        # vectorized equivalent of the per-patch mean loop.
        trimmed = self.data[: new_h * factor, : new_w * factor]
        downsampled = trimmed.reshape(new_h, factor, new_w, factor).mean(axis=(1, 3))

        return DepthMap(
            data=downsampled,
            scale=self.scale,
            shift=self.shift,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

    def to_3d(self, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        """Vectorized back-projection of every valid-depth pixel to 3D."""
        h, w = self.data.shape
        ys, xs = np.mgrid[0:h, 0:w]
        z = self.data

        mask = (z >= self.min_depth) & (z <= self.max_depth)
        x_3d = (xs - cx) * z / fx
        y_3d = (ys - cy) * z / fy

        if not np.any(mask):
            return np.zeros((0, 3))

        return np.stack([x_3d[mask], y_3d[mask], z[mask]], axis=-1)


class DepthEstimator:
    """Monocular depth estimation with a real-model backend and a placeholder.

    Args:
        model: ``"heuristic"`` (default, no extra deps, NOT a real depth
            model — see module docstring) or ``"midas"`` (real MiDaS_small
            inference, requires ``pip install "pyrobovision[depth]"``).
        min_depth: Minimum plausible depth (meters), used for clipping.
        max_depth: Maximum plausible depth (meters), used for clipping.
        scale: Metadata stored on produced ``DepthMap``s (not applied
            automatically).
        device: Torch device string used only when ``model="midas"``.
    """

    def __init__(
        self,
        model: str = "heuristic",
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        scale: float = 0.001,
        device: str = "cpu",
    ):
        if model not in _VALID_MODELS:
            raise ValueError(f"model must be one of {_VALID_MODELS}, got {model!r}")

        self.model = model
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scale = scale
        self.device = device
        self.calibration_matrix = None

        self._midas_model = None
        self._midas_torch = None

    def set_calibration(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self.calibration_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # -- Public API ---------------------------------------------------

    def estimate_depth(self, image: np.ndarray, use_median: bool = True) -> DepthMap:
        if self.model == "midas":
            return self._estimate_depth_midas(image)
        return self._estimate_depth_heuristic(image, use_median=use_median)

    # -- Real model: MiDaS --------------------------------------------

    def _load_midas(self) -> None:
        if self._midas_model is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "model='midas' requires PyTorch. Install it with "
                "`pip install \"pyrobovision[depth]\"` (or `pip install torch`), "
                "or use DepthEstimator(model='heuristic') for a dependency-free "
                "(but non-neural, non-metric) placeholder."
            ) from exc

        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        model.eval()
        model.to(self.device)

        self._midas_torch = torch
        self._midas_model = model

    def _estimate_depth_midas(self, image: np.ndarray) -> DepthMap:
        self._load_midas()
        torch = self._midas_torch

        rgb = self._to_rgb_uint8(image)
        h, w = rgb.shape[:2]

        inp = self._midas_preprocess(rgb)  # (1, 3, 256, 256) tensor
        with torch.no_grad():
            prediction = self._midas_model(inp.to(self.device))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        inverse_depth = prediction.cpu().numpy()

        # MiDaS predicts *relative inverse depth* (larger = closer). Convert
        # to an ordered, positive "depth" in [min_depth, max_depth] so it's
        # drop-in compatible with the rest of the pipeline (to_3d, occupancy
        # grids, etc). This is still RELATIVE depth, not metric meters,
        # unless separately calibrated against a known reference distance.
        inv_min, inv_max = float(inverse_depth.min()), float(inverse_depth.max())
        if inv_max - inv_min < 1e-8:
            normalized = np.zeros_like(inverse_depth)
        else:
            normalized = (inverse_depth - inv_min) / (inv_max - inv_min)

        # Flip so that larger output value = farther away (matches the
        # min_depth/max_depth/to_3d convention used elsewhere in this module).
        depth = self.max_depth - normalized * (self.max_depth - self.min_depth)

        return DepthMap(
            data=np.clip(depth, self.min_depth, self.max_depth),
            scale=self.scale,
            shift=0.0,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

    @staticmethod
    def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            if image.max() <= 1.0 + 1e-6:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return image

    @staticmethod
    def _midas_preprocess(rgb: np.ndarray, size: int = 256):
        """Resize + ImageNet-normalize an RGB uint8 image for MiDaS_small.

        Implemented with PIL + NumPy (no OpenCV dependency) to match the
        preprocessing MiDaS_small expects: bicubic resize to size x size,
        scale to [0, 1], normalize with ImageNet mean/std, CHW, batch dim.
        """
        import torch
        from PIL import Image

        pil_img = Image.fromarray(rgb).resize((size, size), Image.BICUBIC)
        arr = np.asarray(pil_img).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        chw = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0).float()
        return tensor

    # -- Placeholder heuristic ------------------------------------------

    def _estimate_depth_heuristic(self, image: np.ndarray, use_median: bool = True) -> DepthMap:
        """Sobel-edge-based depth PLACEHOLDER.

        This is NOT a depth estimation model. It assigns a slightly closer
        value to edge pixels and a slightly farther value to flat regions,
        then median-filters the result. It has no learned or geometric basis
        and should only be used to exercise downstream code paths, never for
        anything requiring actual depth accuracy.
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        base_depth = 5.0
        edge_mask = self._heuristic_edges(image)

        depth_estimate = np.ones((h, w), dtype=np.float32) * base_depth
        depth_estimate[edge_mask > 0] *= 0.7
        depth_estimate[edge_mask == 0] *= 1.3

        if use_median:
            depth_estimate = _ndi_median_filter(depth_estimate, size=5, mode="nearest")

        return DepthMap(
            data=np.clip(depth_estimate, self.min_depth, self.max_depth),
            scale=self.scale,
            shift=0.0,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

    @staticmethod
    def _heuristic_edges(image: np.ndarray) -> np.ndarray:
        """Vectorized Sobel edge magnitude (via scipy.ndimage), thresholded
        at the median. Replaces a pure-Python nested-loop implementation.
        """
        gray = np.mean(image, axis=2) if image.ndim == 3 else image

        gx = _ndi_sobel(gray, axis=1, mode="reflect")
        gy = _ndi_sobel(gray, axis=0, mode="reflect")
        magnitude = np.hypot(gx, gy)

        return (magnitude > np.percentile(magnitude, 50)).astype(np.float32)

    # -- Shared post-processing ------------------------------------------

    def fuse_with_lidar(
        self, depth_map: DepthMap, lidar_points: np.ndarray, weight: float = 0.5
    ) -> DepthMap:
        fused_depth = depth_map.data.copy()

        if self.calibration_matrix is None:
            return depth_map

        lidar_points = np.asarray(lidar_points)
        if lidar_points.size == 0:
            return depth_map

        fx = self.calibration_matrix[0, 0]
        fy = self.calibration_matrix[1, 1]
        cx = self.calibration_matrix[0, 2]
        cy = self.calibration_matrix[1, 2]

        x_3d, y_3d, z_3d = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
        in_front = z_3d > 0

        x_2d = np.round((x_3d * fx / np.where(in_front, z_3d, 1)) + cx).astype(np.int64)
        y_2d = np.round((y_3d * fy / np.where(in_front, z_3d, 1)) + cy).astype(np.int64)

        h, w = fused_depth.shape
        in_bounds = in_front & (x_2d >= 0) & (x_2d < w) & (y_2d >= 0) & (y_2d < h)

        ys, xs, zs = y_2d[in_bounds], x_2d[in_bounds], z_3d[in_bounds]
        # NumPy fancy-index assignment with repeated (y, x) pairs keeps only
        # the last write per pixel, matching the original loop's behavior.
        fused_depth[ys, xs] = (1 - weight) * fused_depth[ys, xs] + weight * zs

        return DepthMap(
            data=np.clip(fused_depth, self.min_depth, self.max_depth),
            scale=depth_map.scale,
            shift=depth_map.shift,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

    def estimate_uncertainty(self, depth_map: DepthMap) -> np.ndarray:
        """Local standard deviation over a 3x3 neighborhood (vectorized).

        Uses the identity Var(X) = E[X^2] - E[X]^2 with two box-filter passes
        (``scipy.ndimage.uniform_filter``) instead of a per-pixel Python loop
        or per-window callable — both filter passes run as compiled code.
        """
        from scipy.ndimage import uniform_filter

        data = depth_map.data.astype(np.float64)
        mean = uniform_filter(data, size=3, mode="nearest")
        mean_sq = uniform_filter(data * data, size=3, mode="nearest")
        variance = np.clip(mean_sq - mean * mean, 0.0, None)
        return np.sqrt(variance)
