"""Tests for latency-adaptive frame skipping/downsampling (RECHECK-028)."""

import numpy as np
import pytest

from pyrobovision.perception import (
    AdaptiveFrameConfig,
    AdaptiveFrameController,
)


def _config(**kw):
    defaults = dict(target_latency_ms=33.3, ewma_alpha=1.0, max_skip=4)
    defaults.update(kw)
    return AdaptiveFrameConfig(**defaults)


def test_starts_full_rate_and_native_scale():
    c = AdaptiveFrameController(_config())
    assert c.scale == 1.0
    assert c.should_process() is True
    assert c.effective_fps == 1.0


def test_high_latency_triggers_skip_and_downsample():
    c = AdaptiveFrameController(_config())
    # One frame blows well past the budget.
    c.observe(200.0)

    assert c.scale < 1.0
    # At least one frame must now be skipped.
    skipped = [c.should_process() for _ in range(3)]
    assert skipped[0] is False
    assert c.effective_fps < 1.0


def test_skip_count_scales_with_overshoot_and_is_capped():
    c = AdaptiveFrameController(_config(max_skip=4))
    c.observe(1000.0)  # ~30x over a 33.3ms budget

    decisions = [c.should_process() for _ in range(10)]
    n_skipped = decisions.count(False)
    assert 1 <= n_skipped <= 4  # capped by max_skip


def test_latency_within_budget_never_skips():
    c = AdaptiveFrameController(_config())
    for _ in range(20):
        c.observe(10.0)
        assert c.should_process() is True
    assert c.scale == 1.0


def test_scale_recovers_after_load_subsides():
    c = AdaptiveFrameController(_config())
    for _ in range(10):
        c.observe(200.0)
    assert c.scale == pytest.approx(c.config.min_scale)

    for _ in range(20):
        c.observe(1.0)
        c.should_process()
    assert c.scale == pytest.approx(c.config.max_scale)


def test_scale_is_bounded_by_config():
    c = AdaptiveFrameController(_config(min_scale=0.6, max_scale=1.0, scale_step=0.5))
    for _ in range(10):
        c.observe(500.0)
    assert c.config.min_scale <= c.scale <= c.config.max_scale

    for _ in range(10):
        c.observe(0.1)
    assert c.scale == pytest.approx(c.config.max_scale)


def test_downsample_reduces_frame_shape():
    c = AdaptiveFrameController(_config(min_scale=0.5, scale_step=0.5))
    c.observe(500.0)  # scale -> 0.5

    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    out = c.downsample(frame)
    assert out.shape[0] < frame.shape[0]
    assert out.shape[1] < frame.shape[1]
    assert out.shape[2] == 3


def test_downsample_is_noop_at_native_scale():
    c = AdaptiveFrameController(_config())
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    out = c.downsample(frame)
    assert out.shape == frame.shape
    assert np.array_equal(out, frame)


def test_target_size_tracks_scale():
    c = AdaptiveFrameController(_config(min_scale=0.5, scale_step=0.5))
    c.observe(500.0)
    w, h = c.target_size(1920, 1080)
    assert w == int(round(1920 * c.scale))
    assert h == int(round(1080 * c.scale))


def test_decide_is_non_consuming():
    c = AdaptiveFrameController(_config())
    c.observe(500.0)
    d1 = c.decide()
    d2 = c.decide()
    assert d1 == d2
    assert d1.skipping is True
    assert d1.process is False


def test_reset_restores_initial_state():
    c = AdaptiveFrameController(_config())
    for _ in range(5):
        c.observe(500.0)
    c.should_process()
    c.reset()
    assert c.scale == 1.0
    assert c.latency_ms == 0.0
    assert c.effective_fps == 1.0
    assert c.should_process() is True


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        AdaptiveFrameConfig(target_latency_ms=0)
    with pytest.raises(ValueError):
        AdaptiveFrameConfig(ewma_alpha=0)
    with pytest.raises(ValueError):
        AdaptiveFrameConfig(min_scale=0.9, max_scale=0.5)


def test_negative_latency_rejected():
    c = AdaptiveFrameController(_config())
    with pytest.raises(ValueError):
        c.observe(-1.0)
