"""P4 — loader hardening: curriculum + goal-conditioned sampling, windowed video sync."""

import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pyroboframes as prf
from test_loader import CAM, make_dataset


def make_var_dataset(root: str, lengths: list[int]) -> int:
    """A tabular dataset (no video) with per-episode `lengths`; observation.state[i] = [i, 0, 0]."""
    total = sum(lengths)
    os.makedirs(f"{root}/meta/episodes/chunk-000")
    os.makedirs(f"{root}/data/chunk-000")

    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": len(lengths),
        "total_frames": total,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [3]},
            "action": {"dtype": "float32", "shape": [3]},
        },
    }
    with open(f"{root}/meta/info.json", "w") as fh:
        json.dump(info, fh)

    starts, idx = [], 0
    for length in lengths:
        starts.append(idx)
        idx += length
    ends = [s + n for s, n in zip(starts, lengths)]
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array(range(len(lengths)), pa.int64()),
                "length": pa.array(lengths, pa.int64()),
                "dataset_from_index": pa.array(starts, pa.int64()),
                "dataset_to_index": pa.array(ends, pa.int64()),
                "data/chunk_index": pa.array([0] * len(lengths), pa.int64()),
                "data/file_index": pa.array([0] * len(lengths), pa.int64()),
            }
        ),
        f"{root}/meta/episodes/chunk-000/file-000.parquet",
    )

    state = [[float(i), 0.0, 0.0] for i in range(total)]
    action = [[float(i)] * 3 for i in range(total)]
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array(state, pa.list_(pa.float32())),
                "action": pa.array(action, pa.list_(pa.float32())),
            }
        ),
        f"{root}/data/chunk-000/file-000.parquet",
    )
    return total


def test_curriculum_orders_short_episodes_first(tmp_path):
    # episode 0 has 5 frames (0..4), episode 1 has 3 (5..7) — shorter episode 1 must come first.
    make_var_dataset(str(tmp_path), [5, 3])
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    seen = []
    for b in ds.loader(batch_size=4, curriculum=True):
        seen.extend(int(round(x)) for x in b["observation.state"][:, 0])
    assert seen == [5, 6, 7, 0, 1, 2, 3, 4]


def test_goal_conditioned_final(tmp_path):
    make_dataset(str(tmp_path), episodes=2, length=4)  # state[i] = [i, 0, 0]
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    b = next(iter(ds.loader(batch_size=8, shuffle=False, goal="final")))

    assert "observation.state.goal" in b
    # Goal = final frame of the episode: ep0 -> frame 3, ep1 -> frame 7.
    np.testing.assert_array_equal(b["observation.state.goal"][:, 0], [3, 3, 3, 3, 7, 7, 7, 7])
    np.testing.assert_array_equal(b["observation.state"][:, 0], list(range(8)))  # current unchanged


def test_goal_rejects_unsupported_combos(tmp_path):
    make_dataset(str(tmp_path), episodes=2, length=4)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    with pytest.raises(ValueError):
        ds.loader(goal="middle")
    with pytest.raises(ValueError):
        ds.loader(goal="final", num_workers=2)


def test_windowed_video_sync(tmp_path):
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    make_dataset(str(tmp_path), episodes=1, length=10, with_video=True)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    loader = ds.loader(
        batch_size=4,
        shuffle=False,
        cameras=[CAM],
        delta_timestamps={CAM: [-1 / 30, 0.0]},  # previous frame + current
    )
    b = next(iter(loader))
    frames = b[CAM]
    assert frames.ndim == 5  # [batch, steps, H, W, 3]
    assert frames.shape[0] == 4
    assert frames.shape[1] == 2  # two timesteps
    assert frames.shape[4] == 3
