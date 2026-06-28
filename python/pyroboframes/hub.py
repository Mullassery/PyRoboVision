"""Hugging Face Hub interop: download a ``LeRobotDataset`` so it can be opened locally.

Supports full-download (``snapshot_download``) and partial-streaming (download only accessed
episodes on-demand). Network-dependent, so it isn't exercised by the offline test suite.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi


def download_lerobot_dataset(
    repo_id: str,
    local_dir: str | None = None,
    revision: str | None = None,
    episodes: list[int] | None = None,
) -> str:
    """Download a LeRobot dataset from the Hugging Face Hub and return its local path.

    By default, downloads the entire dataset (``snapshot_download``). If ``episodes`` is
    specified, downloads only metadata + the given episode indices, streaming others on-demand.

    Full download example::

        import pyroboframes as prf
        local_path = prf.download_lerobot_dataset("lerobot/aloha_mobile_cabinet")
        ds = prf.RoboFrameDataset.from_path(local_path)

    Partial download with on-demand streaming::

        # Download only episodes 0 and 1; others stream when accessed
        local_path = prf.download_lerobot_dataset(
            "lerobot/aloha_mobile_cabinet",
            episodes=[0, 1],
        )
        ds = prf.RoboFrameDataset.from_path(local_path)
        # Accessing episode 2 will download it on-demand
        loader = ds.loader(episodes=[0, 1, 2])

    Args:
        repo_id: Hugging Face dataset repo (e.g., "lerobot/aloha_mobile_cabinet")
        local_dir: Cache directory (default: HF cache at `~/.cache/huggingface/datasets/`)
        revision: Git revision (default: "main")
        episodes: List of episode indices to pre-download; others stream on-demand. None = full
                  dataset. Requires `huggingface_hub >= 0.21.0`.

    Returns:
        Local path to the dataset (can be opened with :class:`pyroboframes.RoboFrameDataset`).
        Requires the optional ``huggingface_hub`` package (``pip install huggingface_hub``).
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "download_lerobot_dataset requires `huggingface_hub` "
            "(pip install huggingface_hub)"
        ) from exc

    if episodes is None:
        return snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            revision=revision,
        )

    return _download_lerobot_partial(
        repo_id=repo_id,
        episodes=episodes,
        local_dir=local_dir,
        revision=revision,
    )


def _download_lerobot_partial(
    repo_id: str,
    episodes: list[int],
    local_dir: str | None = None,
    revision: str | None = None,
) -> str:
    """Download only metadata + specific episodes from a LeRobot dataset, returning the root dir.

    The dataset root is populated with:
    - meta/ (always)
    - data/chunk-*/file-*.parquet (only those containing the requested episodes)
    - videos/ (only those containing the requested episodes)

    Other episodes are fetched on-demand when accessed via the dataloader.
    """
    from huggingface_hub import hf_hub_download

    revision = revision or "main"

    if local_dir is None:
        from huggingface_hub import HfFolder
        cache_home = HfFolder.home()
        local_dir = os.path.join(
            cache_home, "datasets", repo_id.replace("/", "--"), revision
        )

    os.makedirs(local_dir, exist_ok=True)

    hf_hub_download(
        repo_id=repo_id,
        filename="meta/info.json",
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
    )

    meta_episodes = hf_hub_download(
        repo_id=repo_id,
        filename="meta/episodes/chunk-000/file-000.parquet",
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
    )

    import pyarrow.parquet as pq

    ep_table = pq.read_table(meta_episodes)
    ep_df = ep_table.to_pandas()

    for ep_idx in episodes:
        if ep_idx < 0 or ep_idx >= len(ep_df):
            continue

        ep_row = ep_df.iloc[ep_idx]
        data_chunk = int(ep_row["data/chunk_index"])
        data_file = int(ep_row["data/file_index"])

        data_path = f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet"
        hf_hub_download(
            repo_id=repo_id,
            filename=data_path,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
        )

        for cam_col in ep_df.columns:
            if cam_col.startswith("videos/") and cam_col.endswith("/chunk_index"):
                cam_key = cam_col.replace("/chunk_index", "")
                vid_chunk = int(ep_row[f"{cam_key}/chunk_index"])
                vid_file = int(ep_row[f"{cam_key}/file_index"])
                vid_path = f"videos/{cam_key}/chunk-{vid_chunk:03d}/file-{vid_file:03d}.mp4"

                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=vid_path,
                        repo_type="dataset",
                        revision=revision,
                        local_dir=local_dir,
                    )
                except Exception:
                    pass

    return local_dir
