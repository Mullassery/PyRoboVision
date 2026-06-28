"""LeRobot write-back: export tabular feature arrays as a `LeRobotDataset v3.0` directory.

This is the inverse of the reader: given per-feature arrays and episode boundaries, it writes the
v3.0 layout PyRoboFrames (and LeRobot) read — ``meta/info.json``, ``meta/episodes/*.parquet``,
``data/*.parquet`` (fixed-size ``float32`` lists), and ``meta/stats.json``. Round-trips through
:class:`pyroboframes.RoboFrameDataset`. Video features are out of scope here (tabular only).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _feature_stats(arr: np.ndarray) -> dict:
    """Per-dimension mean/std/min/max + scalar count for a ``[N, D]`` feature."""
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "count": int(arr.shape[0]),
    }


def write_lerobot_dataset(
    path: str,
    features: dict[str, np.ndarray],
    episode_lengths: list[int],
    fps: float = 30.0,
    robot_type: str | None = None,
    video_codec: str = "h264",
    video_profile: str | None = None,
) -> None:
    """Write a tabular LeRobotDataset v3.0 at ``path``.

    ``features`` maps each feature name (e.g. ``observation.state``, ``action``) to a ``[N, D]``
    array; all features must share ``N``, the total frame count. ``episode_lengths`` partitions the
    ``N`` frames into episodes (in order) and must sum to ``N``.

    Args:
        video_codec: Video codec for video features (``"h264"`` [default], ``"hevc"``, ``"av1"``).
        video_profile: Codec profile (e.g., ``"main"`` for HEVC). None = default per codec.

    Note: Currently used for metadata only; actual video encoding via external FFmpeg CLI.
    """
    if not features:
        raise ValueError("at least one feature is required")
    if video_codec not in ("h264", "hevc", "av1"):
        raise ValueError(f"video_codec must be 'h264', 'hevc', or 'av1', got {video_codec!r}")

    arrays = {name: np.asarray(v, dtype=np.float32) for name, v in features.items()}
    for name, arr in arrays.items():
        if arr.ndim != 2:
            raise ValueError(f"feature {name!r} must be 2-D [N, D], got shape {arr.shape}")
    total = next(iter(arrays.values())).shape[0]
    if any(arr.shape[0] != total for arr in arrays.values()):
        raise ValueError("all features must have the same number of frames")
    if sum(episode_lengths) != total:
        raise ValueError(
            f"episode_lengths sum ({sum(episode_lengths)}) != total frames ({total})"
        )

    os.makedirs(os.path.join(path, "meta", "episodes", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(path, "data", "chunk-000"), exist_ok=True)

    _write_data(path, arrays)
    _write_episodes(path, episode_lengths)
    _write_info(path, arrays, episode_lengths, total, fps, robot_type, video_codec, video_profile)
    _write_stats(path, arrays)


def _write_data(path: str, arrays: dict[str, np.ndarray]) -> None:
    columns = {}
    for name, arr in arrays.items():
        dim = arr.shape[1]
        values = pa.array(arr.reshape(-1), type=pa.float32())
        columns[name] = pa.FixedSizeListArray.from_arrays(values, dim)
    table = pa.table(columns)
    pq.write_table(table, os.path.join(path, "data", "chunk-000", "file-000.parquet"))


def _write_episodes(path: str, episode_lengths: list[int]) -> None:
    starts, idx = [], 0
    for length in episode_lengths:
        starts.append(idx)
        idx += length
    ends = [s + length for s, length in zip(starts, episode_lengths)]
    n = len(episode_lengths)
    table = pa.table(
        {
            "episode_index": pa.array(range(n), pa.int64()),
            "length": pa.array(episode_lengths, pa.int64()),
            "dataset_from_index": pa.array(starts, pa.int64()),
            "dataset_to_index": pa.array(ends, pa.int64()),
            "data/chunk_index": pa.array([0] * n, pa.int64()),
            "data/file_index": pa.array([0] * n, pa.int64()),
        }
    )
    pq.write_table(
        table, os.path.join(path, "meta", "episodes", "chunk-000", "file-000.parquet")
    )


def _write_info(
    path: str,
    arrays: dict[str, np.ndarray],
    episode_lengths: list[int],
    total: int,
    fps: float,
    robot_type: str | None,
    video_codec: str = "h264",
    video_profile: str | None = None,
) -> None:
    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "fps": fps,
        "total_episodes": len(episode_lengths),
        "total_frames": total,
        "chunks_size": 1000,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "video_codec": video_codec,
    }
    if video_profile:
        info["video_profile"] = video_profile
    info["features"] = {
        name: {"dtype": "float32", "shape": [arr.shape[1]]}
        for name, arr in arrays.items()
    }
    with open(os.path.join(path, "meta", "info.json"), "w") as fh:
        json.dump(info, fh, indent=2)


def _write_stats(path: str, arrays: dict[str, np.ndarray]) -> None:
    stats = {name: _feature_stats(arr) for name, arr in arrays.items()}
    with open(os.path.join(path, "meta", "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
