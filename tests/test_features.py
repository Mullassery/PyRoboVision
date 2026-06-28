"""Tests for the v0.1.x quick-win additions: dataset statistics (meta/stats.json),
deterministic train/val split, and loader checkpoint/resume (position + seek).
"""

import json

import numpy as np
import pytest

import pyroboframes as prf

from test_loader import make_dataset


def _write_stats(root: str):
    stats = {
        "observation.state": {
            "mean": [1.0, 2.0, 3.0],
            "std": [0.5, 0.5, 0.5],
            "min": [0.0, 0.0, 0.0],
            "max": [9.0, 9.0, 9.0],
            "count": 100,
        },
        "action": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
    }
    with open(f"{root}/meta/stats.json", "w") as fh:
        json.dump(stats, fh)


def test_stats_absent_is_none(tmp_path):
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    assert ds.stats() is None


def test_stats_parsed(tmp_path):
    make_dataset(str(tmp_path))
    _write_stats(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    stats = ds.stats()
    assert stats is not None
    s = stats["observation.state"]
    assert s["mean"] == [1.0, 2.0, 3.0]
    assert s["max"] == [9.0, 9.0, 9.0]
    assert s["count"] == 100
    assert stats["action"]["std"] == [1.0, 1.0, 1.0]
    assert stats["action"]["count"] is None


def test_train_val_split_partitions_episodes(tmp_path):
    make_dataset(str(tmp_path), episodes=10, length=5)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    train, val = ds.train_val_split(val_fraction=0.2, seed=0)
    assert len(val) == 2
    assert len(train) == 8
    assert sorted(train + val) == list(range(10))
    # Deterministic for a fixed seed.
    assert (train, val) == ds.train_val_split(val_fraction=0.2, seed=0)


def test_episodes_metadata(tmp_path):
    make_dataset(str(tmp_path), episodes=3, length=20)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    eps = ds.episodes()
    assert len(eps) == 3
    assert eps[0] == {"episode_index": 0, "length": 20, "from_index": 0, "to_index": 20}
    assert eps[2]["from_index"] == 40 and eps[2]["to_index"] == 60


def test_loader_normalize(tmp_path):
    make_dataset(str(tmp_path))
    # state[i] = [i,0,0]; with mean [10,0,0] std [2,1,1] -> ((i-10)/2, 0, 0).
    with open(f"{tmp_path}/meta/stats.json", "w") as fh:
        json.dump(
            {
                "observation.state": {"mean": [10.0, 0.0, 0.0], "std": [2.0, 1.0, 1.0]},
                "action": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
            },
            fh,
        )
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    loader = ds.loader(batch_size=10, shuffle=False, normalize=["observation.state"])
    b0 = next(iter(loader))
    np.testing.assert_allclose(b0["observation.state"][5], [(5 - 10) / 2, 0.0, 0.0])
    np.testing.assert_array_equal(b0["action"][5], [5.0, 5.0, 5.0])  # untouched


def test_prefetch_matches_sync(tmp_path):
    total = make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    def frames(**kw):
        seen = []
        for b in ds.loader(batch_size=8, shuffle=False, **kw):
            seen.extend(int(round(x)) for x in b["observation.state"][:, 0])
        return seen

    sync = frames()
    pre = frames(num_workers=3, prefetch=2)  # off-GIL prefetch pipeline
    assert sync == pre == list(range(total))  # same order, every frame once
    assert len(ds.loader(batch_size=8, num_workers=2)) == len(ds.loader(batch_size=8))


def test_resolve_device(monkeypatch):
    # Explicit device passes through; invalid raises.
    assert prf.resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        prf.resolve_device("tpu")
    # Env override is honored for "auto".
    monkeypatch.setenv("PYROBOFRAMES_DEVICE", "cuda")
    assert prf.resolve_device("auto") == "cuda"
    monkeypatch.delenv("PYROBOFRAMES_DEVICE", raising=False)
    # With nothing available, auto falls back to a valid backend.
    assert prf.resolve_device("auto") in prf.backend.VALID_DEVICES


def test_transform_backend_resolution():
    """Verify backend resolution fallback chain: CV-CUDA → MLX → Torch → NumPy."""
    from pyroboframes import transforms as T

    # Test that resolve_transform_backend follows the correct fallback order
    backend = T.resolve_transform_backend("auto")
    assert backend in T.TRANSFORM_BACKENDS

    # NumPy should always be available (last in chain)
    backend = T.resolve_transform_backend("numpy")
    assert backend == "numpy"

    # Invalid backend raises ValueError
    with pytest.raises(ValueError):
        T.resolve_transform_backend("invalid")


def test_transforms_shapes_and_values():
    from pyroboframes import transforms as T

    x = np.zeros((2, 8, 6, 3), dtype=np.uint8)
    x[:, :, :, 0] = 255  # red channel maxed

    resized = T.Resize(4, 3)(x)
    assert resized.shape == (2, 4, 3, 3)

    cropped = T.CenterCrop(4, 4)(x)
    assert cropped.shape == (2, 4, 4, 3)

    norm = T.Normalize(mean=[0.5, 0.0, 0.0], std=[0.5, 1.0, 1.0])(x)
    assert norm.dtype == np.float32
    # red: (1.0 - 0.5)/0.5 = 1.0 ; green/blue: (0 - 0)/1 = 0
    np.testing.assert_allclose(norm[0, 0, 0], [1.0, 0.0, 0.0])

    composed = T.Compose([T.Resize(4, 4), T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])(x)
    assert composed.shape == (2, 4, 4, 3) and composed.dtype == np.float32


def test_bilinear_resize_and_augments():
    from pyroboframes import transforms as T

    x = np.zeros((2, 8, 6, 3), dtype=np.uint8)
    x[:, :, :, 0] = 200

    # Bilinear returns float32 and the requested shape; a flat image stays ~constant.
    r = T.Resize(4, 3)(x)  # bilinear default
    assert r.shape == (2, 4, 3, 3) and r.dtype == np.float32
    np.testing.assert_allclose(r[..., 0], 200.0, atol=1e-3)
    assert T.Resize(4, 3, interpolation="nearest")(x).dtype == np.uint8

    # Flip: with a left/right-asymmetric image, flipping must change it; seeded -> deterministic.
    g = np.tile(np.arange(6, dtype=np.uint8), (2, 8, 1))[..., None].repeat(3, axis=-1)
    flipped = T.RandomHorizontalFlip(p=1.0, seed=0)(g)
    np.testing.assert_array_equal(flipped, g[:, :, ::-1, :])
    np.testing.assert_array_equal(T.RandomHorizontalFlip(p=0.0)(g), g)  # never flip

    cropped = T.RandomCrop(4, 4, seed=1)(x)
    assert cropped.shape == (2, 4, 4, 3) and cropped.dtype == np.uint8

    jit = T.ColorJitter(brightness=0.5, seed=2)(x)
    assert jit.dtype == np.uint8 and jit.max() <= 255


def test_dataloader_applies_transforms_on_cpu():
    from pyroboframes import transforms as T

    # Fake inner loader: two batches, each with an image + a state vector.
    def fake_inner():
        for _ in range(2):
            yield {
                "observation.images.top": np.zeros((3, 8, 6, 3), dtype=np.uint8),
                "observation.state": np.ones((3, 4), dtype=np.float32),
            }

    class _Inner:
        def __iter__(self):
            return fake_inner()

        def __len__(self):
            return 2

    seen = []
    loader = prf.DataLoader(
        _Inner(),
        transforms=T.Resize(4, 4, interpolation="nearest"),
        device="cpu",
        on_batch=lambda i, b, dt: seen.append((i, dt)),
    )
    assert len(loader) == 2
    batches = list(loader)
    assert batches[0]["observation.images.top"].shape == (3, 4, 4, 3)  # transformed
    assert batches[0]["observation.state"].shape == (3, 4)  # untouched
    # Profiling hook fired per batch; stats accumulated.
    assert [i for i, _ in seen] == [0, 1]
    assert loader.stats["batches"] == 2 and loader.stats["frames"] == 6
    assert loader.stats["frames_per_s"] > 0


def test_jax_output(tmp_path):
    jax = pytest.importorskip("jax")  # skip cleanly if JAX isn't installed
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    b = next(iter(ds.loader(batch_size=8, shuffle=False, output="jax")))
    assert isinstance(b["observation.state"], jax.Array)
    assert b["observation.state"].shape == (8, 3)


def test_balanced_sampling_smoke(tmp_path):
    total = make_dataset(str(tmp_path), episodes=4, length=25)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    loader = ds.loader(batch_size=10, balanced=True, seed=0)
    seen = []
    for b in loader:
        seen.extend(int(round(x)) for x in b["observation.state"][:, 0])
    assert len(seen) == total  # one epoch's worth of draws
    assert all(0 <= f < total for f in seen)
    # Balanced sampling is with replacement, so it should touch every episode.
    episodes_hit = {f // 25 for f in seen}
    assert episodes_hit == {0, 1, 2, 3}


def test_episode_chunking_keeps_sequences_contiguous(tmp_path):
    # 3 episodes x 8 frames; state[i, 0] == global frame index i. chunk_size=4 divides 8 evenly,
    # so the epoch is exactly 6 shuffled chunks of 4 contiguous in-episode frames.
    make_dataset(str(tmp_path), episodes=3, length=8)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    loader = ds.loader(batch_size=4, shuffle=True, seed=0, chunk_size=4)

    seen = []
    for b in loader:
        seen.extend(int(round(x)) for x in b["observation.state"][:, 0])

    assert sorted(seen) == list(range(24))  # every frame exactly once
    for g in (seen[i : i + 4] for i in range(0, 24, 4)):
        assert g == list(range(g[0], g[0] + 4))  # ascending + contiguous
        assert g[0] // 8 == g[-1] // 8  # never crosses an episode boundary
    assert seen != list(range(24))  # chunks were actually shuffled


def test_episode_chunking_respects_episode_filter(tmp_path):
    make_dataset(str(tmp_path), episodes=4, length=10)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    # Restrict to episodes 1 and 3, episode-chunked.
    loader = ds.loader(batch_size=5, shuffle=True, seed=1, chunk_size=5, episodes=[1, 3])
    seen = sorted(int(round(x)) for b in loader for x in b["observation.state"][:, 0])
    expected = sorted(list(range(10, 20)) + list(range(30, 40)))
    assert seen == expected


def test_mlx_sequence_batching(tmp_path):
    mx = pytest.importorskip("mlx.core")  # skip cleanly if MLX isn't installed
    make_dataset(str(tmp_path), episodes=2, length=20)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    # Temporal window (3 steps) + episode chunking -> [batch, steps, dim] sequences as MLX arrays.
    loader = ds.loader(
        batch_size=4,
        shuffle=True,
        seed=0,
        chunk_size=5,
        delta_timestamps={"observation.state": [-0.2, -0.1, 0.0]},
        output="mlx",
    )
    b = next(iter(loader))
    seq = b["observation.state"]
    assert isinstance(seq, mx.array)
    assert seq.shape == (4, 3, 3)  # [batch, num_steps, feature_dim]


def test_loader_filters_to_split_episodes(tmp_path):
    # 4 episodes x 25 frames; episode e owns global frames [e*25, (e+1)*25).
    make_dataset(str(tmp_path), episodes=4, length=25)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    train, val = ds.train_val_split(val_fraction=0.5, seed=1)
    assert len(train) == 2 and len(val) == 2

    def frames(loader):
        seen = []
        for b in loader:
            seen.extend(int(round(x)) for x in b["observation.state"][:, 0])
        return sorted(seen)

    def expected(eps):
        return sorted(f for e in eps for f in range(e * 25, (e + 1) * 25))

    train_frames = frames(ds.loader(batch_size=8, shuffle=True, seed=3, episodes=train))
    val_frames = frames(ds.loader(batch_size=8, shuffle=False, episodes=val))

    assert train_frames == expected(train)
    assert val_frames == expected(val)
    # No leakage: train and val frames are disjoint and together cover everything.
    assert set(train_frames).isdisjoint(val_frames)
    assert sorted(train_frames + val_frames) == list(range(100))


def test_loader_position_and_seek_resume(tmp_path):
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    # Consume two batches, remember where we are.
    loader = ds.loader(batch_size=10, shuffle=False)
    assert loader.position == 0
    it = iter(loader)
    next(it)
    next(it)
    assert loader.position == 20

    # Resume a fresh (identical) loader at the saved position; remaining frames must match.
    resumed = ds.loader(batch_size=10, shuffle=False)
    resumed.seek(20)
    assert resumed.position == 20
    remaining = []
    for b in resumed:
        remaining.extend(int(round(x)) for x in b["observation.state"][:, 0])
    assert remaining == list(range(20, 100))  # frames 0..19 were skipped

    # Seek past the end is clamped (no error, empty iteration).
    done = ds.loader(batch_size=10, shuffle=False)
    done.seek(10_000)
    assert list(done) == []
