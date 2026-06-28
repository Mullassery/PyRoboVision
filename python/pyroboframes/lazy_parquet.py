"""Lazy row-group-level Parquet streaming for large datasets.

Avoids loading entire shards into memory by reading Parquet files row-group by row-group.
Useful for datasets larger than available RAM.

```python
reader = LazyParquetReader(path)
for batch in reader.iter_row_groups(batch_size=1000):
    # batch is a pyarrow.Table spanning one or more row-groups
    process(batch)
```
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pyarrow as pa


class LazyParquetReader:
    """Stream a Parquet file row-group by row-group, avoiding full-shard loads."""

    def __init__(self, path: str | Path):
        """Initialize lazy reader for a Parquet file.

        Args:
            path: Path to .parquet file
        """
        self.path = Path(path)
        self._parquet_file = pq.ParquetFile(str(self.path))

    @property
    def schema(self) -> pa.Schema:
        """Arrow schema of the Parquet file."""
        return self._parquet_file.schema_arrow

    @property
    def num_rows(self) -> int:
        """Total row count."""
        return self._parquet_file.metadata.num_rows

    @property
    def num_row_groups(self) -> int:
        """Number of row-groups in the file."""
        return self._parquet_file.num_row_groups

    def iter_row_groups(
        self, columns: list[str] | None = None, batch_size: int | None = None
    ) -> Iterator[pa.Table]:
        """Iterate row-groups as Arrow tables.

        Args:
            columns: Column names to read (default: all)
            batch_size: If set, accumulate row-groups until total rows >= batch_size

        Yields:
            pyarrow.Table for each row-group (or batch of row-groups)
        """
        if batch_size is None:
            for i in range(self.num_row_groups):
                yield self._parquet_file.read_row_group(i, columns=columns)
        else:
            batch = None
            for i in range(self.num_row_groups):
                rg = self._parquet_file.read_row_group(i, columns=columns)
                batch = rg if batch is None else batch.append(rg)
                if batch.num_rows >= batch_size:
                    yield batch
                    batch = None
            if batch is not None and batch.num_rows > 0:
                yield batch

    def read_all(self, columns: list[str] | None = None) -> pa.Table:
        """Read the entire file into memory (use only if it fits)."""
        return self._parquet_file.read(columns=columns)


class LazyDataFrameShards:
    """Manage multiple Parquet shards with row-group-level streaming."""

    def __init__(self, shard_paths: list[str | Path]):
        """Initialize lazy readers for multiple shards.

        Args:
            shard_paths: List of paths to .parquet files
        """
        self.shard_paths = [Path(p) for p in shard_paths]
        self.readers = [LazyParquetReader(p) for p in self.shard_paths]

    def iter_all_row_groups(
        self, columns: list[str] | None = None, batch_size: int | None = None
    ) -> Iterator[tuple[int, pa.Table]]:
        """Iterate all row-groups across all shards.

        Yields:
            (shard_index, table) tuples
        """
        for shard_idx, reader in enumerate(self.readers):
            for table in reader.iter_row_groups(columns=columns, batch_size=batch_size):
                yield shard_idx, table

    def total_rows(self) -> int:
        """Sum of rows across all shards."""
        return sum(r.num_rows for r in self.readers)
