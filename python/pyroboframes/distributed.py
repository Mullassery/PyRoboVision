"""Distributed data loading for multi-GPU and multi-machine training.

Supports PyTorch distributed sampler for synchronized loading across workers.

```python
import torch.distributed as dist
from pyroboframes.distributed import DistributedLoader

# Initialize distributed backend (via torch.distributed.launch)
dist.init_process_group("nccl")

# Create distributed loader
ds = prf.RoboFrameDataset.from_path("…")
loader = DistributedLoader(
    ds,
    batch_size=32,
    world_size=dist.get_world_size(),
    rank=dist.get_rank(),
    num_workers=4,
)

# Each worker loads different episodes; synchronized across ranks
for batch in loader:
    # All workers yield batches in lockstep
    ...
```
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from ._core import RoboFrameDataset, Loader


class DistributedSampler:
    """Episode sampler for distributed training.

    Divides episodes evenly across workers (no overlap) and supports
    synchronized shuffling and epoch-based re-seeding.
    """

    def __init__(
        self,
        num_episodes: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """Initialize sampler.

        Args:
            num_episodes: Total number of episodes
            num_replicas: Number of distributed processes (world_size)
            rank: Rank of current process (0 to num_replicas-1)
            shuffle: Whether to shuffle episodes
            seed: Random seed (for reproducibility)
            drop_last: Drop last incomplete batch
        """
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, should be in the interval [0, {num_replicas})"
            )

        self.num_episodes = num_episodes
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Calculate how many episodes per worker
        if self.drop_last and self.num_episodes % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (self.num_episodes - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(self.num_episodes / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """Yield episode indices for this worker."""
        import numpy as np

        if self.shuffle:
            # Seeded shuffle for reproducibility
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = np.arange(self.num_episodes)
            rng.shuffle(indices)
            indices = indices.tolist()
        else:
            indices = list(range(self.num_episodes))

        # Pad or trim to make divisible by num_replicas
        if len(indices) < self.total_size:
            # Pad with random indices
            padding = self.total_size - len(indices)
            indices += [indices[i % len(indices)] for i in range(padding)]
        elif len(indices) > self.total_size:
            indices = indices[: self.total_size]

        # Partition by rank
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]

        return iter(indices)

    def __len__(self) -> int:
        """Number of episodes for this worker."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible shuffling.

        Call this at the start of each epoch to ensure different shuffle
        order across epochs while maintaining reproducibility.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch


class DistributedLoader:
    """Wrapper around RoboFrameDataset for distributed training.

    Automatically handles episode partitioning and synchronized loading
    across multiple workers/machines.
    """

    def __init__(
        self,
        dataset: RoboFrameDataset,
        batch_size: int,
        world_size: int,
        rank: int,
        num_workers: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        **loader_kwargs: Any,
    ):
        """Initialize distributed loader.

        Args:
            dataset: RoboFrameDataset to load from
            batch_size: Batch size per worker
            world_size: Total number of distributed processes
            rank: Rank of current process
            num_workers: Number of prefetch workers
            shuffle: Whether to shuffle episodes
            seed: Random seed
            drop_last: Drop last incomplete batch
            **loader_kwargs: Additional kwargs passed to dataset.loader()
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.loader_kwargs = loader_kwargs
        self.epoch = 0

        # Create sampler
        self.sampler = DistributedSampler(
            num_episodes=dataset.num_episodes(),
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def __iter__(self):
        """Iterate batches for this worker."""
        # Set sampler epoch for reproducible shuffling
        self.sampler.set_epoch(self.epoch)

        # Get episodes for this worker
        episodes = list(self.sampler)

        # Create loader with these episodes
        loader = self.dataset.loader(
            episodes=episodes,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **self.loader_kwargs,
        )

        return iter(loader)

    def __len__(self) -> int:
        """Number of batches for this worker."""
        num_episodes = len(self.sampler)
        # Approximate batches (true count depends on episode length)
        avg_episode_length = self.dataset.total_frames() / max(1, self.dataset.num_episodes())
        total_frames = num_episodes * avg_episode_length
        return int(math.ceil(total_frames / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible shuffling.

        Call at the start of each epoch.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch
        self.sampler.set_epoch(epoch)
