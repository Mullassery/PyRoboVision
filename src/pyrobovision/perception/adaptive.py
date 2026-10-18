"""Latency-adaptive frame skipping and downsampling for the perception loop.

RECHECK-028: the perception/tracking loop had no mechanism to shed load when
per-frame processing latency rose above the frame budget, so a slow detector
or an overloaded CPU would silently fall behind the camera's frame rate and
starve the control loop. ``AdaptiveFrameController`` watches recent
processing latency (an EWMA) and, when the budget is exceeded, requests that
subsequent frames be skipped (up to ``max_skip``) and that frames be
downsampled (down to ``min_scale``); when latency recovers it eases both
back toward full rate/resolution.

The controller is deliberately I/O-free and deterministic: the caller feeds
it measured latencies and asks it what to do with the next frame, which
makes the policy unit-testable without a real camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptiveFrameConfig:
    """Tuning for :class:`AdaptiveFrameController`.

    Attributes:
        target_latency_ms: Per-frame processing budget. Latency above this
            triggers skipping + downsampling; below ``recovery_latency_ms``
            eases back toward full rate.
        recovery_latency_ms: Latency at/below which the controller starts
            restoring resolution and skipping fewer frames. Defaults to
            ``0.8 * target_latency_ms`` when ``None``.
        ewma_alpha: Smoothing factor (0..1] for the latency EWMA. Higher is
            more reactive, lower is smoother.
        max_skip: Maximum consecutive frames the controller will ask to skip.
        min_scale / max_scale: Bounds on the resolution scale (1.0 = native).
        scale_step: Amount the scale is reduced/increased per decision.
        skip_ratio_cap: Fraction of frames it is willing to skip at most,
            used to bound the skip count derived from the latency overshoot.
    """

    target_latency_ms: float = 33.3  # ~30 FPS
    recovery_latency_ms: float | None = None
    ewma_alpha: float = 0.2
    max_skip: int = 4
    min_scale: float = 0.5
    max_scale: float = 1.0
    scale_step: float = 0.1
    skip_ratio_cap: float = 0.8

    def __post_init__(self) -> None:
        if self.target_latency_ms <= 0:
            raise ValueError("target_latency_ms must be positive")
        if not (0.0 < self.ewma_alpha <= 1.0):
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.max_skip < 0:
            raise ValueError("max_skip must be >= 0")
        if not (0.0 < self.min_scale <= self.max_scale <= 1.0):
            raise ValueError("require 0 < min_scale <= max_scale <= 1")
        if self.scale_step <= 0:
            raise ValueError("scale_step must be positive")
        if self.recovery_latency_ms is None:
            self.recovery_latency_ms = 0.8 * self.target_latency_ms


@dataclass
class FrameDecision:
    """What the controller wants the caller to do with the next frame."""

    process: bool
    scale: float
    latency_ms: float  # the EWMA used for this decision
    skipping: bool


class AdaptiveFrameController:
    """Decide, per frame, whether to skip and at what resolution to process."""

    def __init__(self, config: AdaptiveFrameConfig | None = None):
        self.config = config or AdaptiveFrameConfig()
        self._ewma_ms: float | None = None
        self._scale: float = self.config.max_scale
        self._skip_remaining: int = 0
        self._frames_seen: int = 0
        self._frames_skipped: int = 0

    # -- observation -----------------------------------------------------
    def observe(self, latency_ms: float) -> None:
        """Feed the measured processing latency of the frame just handled."""
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        cfg = self.config
        if self._ewma_ms is None:
            self._ewma_ms = latency_ms
        else:
            self._ewma_ms = (
                cfg.ewma_alpha * latency_ms + (1.0 - cfg.ewma_alpha) * self._ewma_ms
            )

        if self._ewma_ms > cfg.target_latency_ms:
            self._apply_backpressure()
        elif self._ewma_ms <= cfg.recovery_latency_ms:
            self._relax()

    def _apply_backpressure(self) -> None:
        cfg = self.config
        overshoot = self._ewma_ms / cfg.target_latency_ms
        # 1 skip as soon as the budget is exceeded, scaling with the
        # overshoot, capped by max_skip and by the fraction of frames the
        # controller is willing to drop.
        raw = int(overshoot)  # >= 1 while over budget
        cap = min(cfg.max_skip, int(cfg.skip_ratio_cap / (1.0 - cfg.skip_ratio_cap) + 1))
        self._skip_remaining = max(self._skip_remaining, min(raw, cap))
        self._scale = max(cfg.min_scale, self._scale - cfg.scale_step)

    def _relax(self) -> None:
        cfg = self.config
        self._scale = min(cfg.max_scale, self._scale + cfg.scale_step)
        if self._skip_remaining > 0:
            self._skip_remaining -= 1

    # -- decisions -------------------------------------------------------
    def should_process(self) -> bool:
        """Consume the next frame's decision.

        Returns ``False`` (skip the frame) while the skip budget from
        backpressure is non-zero, ``True`` otherwise. Each call advances the
        frame counter; skipped frames still let the ``scale`` shrink so the
        frames that *are* processed are cheaper.
        """
        self._frames_seen += 1
        if self._skip_remaining > 0:
            self._skip_remaining -= 1
            self._frames_skipped += 1
            return False
        return True

    def decide(self) -> FrameDecision:
        """Non-consuming view of the pending decision (no counters change)."""
        return FrameDecision(
            process=self._skip_remaining == 0,
            scale=self._scale,
            latency_ms=self._ewma_ms if self._ewma_ms is not None else 0.0,
            skipping=self._skip_remaining > 0,
        )

    # -- helpers ---------------------------------------------------------
    def target_size(self, width: int, height: int) -> tuple[int, int]:
        """Pixel size to process the next frame at, given the current scale."""
        w = max(1, int(round(width * self._scale)))
        h = max(1, int(round(height * self._scale)))
        return w, h

    def downsample(self, frame: np.ndarray) -> np.ndarray:
        """Downsample ``frame`` to the current scale.

        Uses OpenCV's ``resize`` (area interpolation) when available;
        otherwise falls back to integer-stride decimation so the helper has
        no hard dependency beyond NumPy.
        """
        if self._scale >= 1.0:
            return frame
        h, w = frame.shape[:2]
        tw, th = self.target_size(w, h)
        try:
            import cv2  # type: ignore

            interp = cv2.INTER_AREA if (tw < w or th < h) else cv2.INTER_LINEAR
            return cv2.resize(frame, (tw, th), interpolation=interp)
        except Exception:
            step_y = max(1, int(round(1.0 / self._scale)))
            sub = frame[::step_y, ::step_y]
            return sub.copy()

    def reset(self) -> None:
        self._ewma_ms = None
        self._scale = self.config.max_scale
        self._skip_remaining = 0
        self._frames_seen = 0
        self._frames_skipped = 0

    # -- introspection ---------------------------------------------------
    @property
    def scale(self) -> float:
        return self._scale

    @property
    def latency_ms(self) -> float:
        return self._ewma_ms if self._ewma_ms is not None else 0.0

    @property
    def effective_fps(self) -> float:
        """Fraction of observed frames that were actually processed (0..1)."""
        if self._frames_seen == 0:
            return 1.0
        return 1.0 - (self._frames_skipped / self._frames_seen)
