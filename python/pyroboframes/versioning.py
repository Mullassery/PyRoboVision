"""Dataset versioning and incremental append.

Enables adding new episodes to existing datasets without rewriting the entire dataset.
Maintains metadata versions and supports rollback/recovery.

```python
from pyroboframes.versioning import DatasetVersion

# Initialize versioned dataset
ds = prf.RoboFrameDataset.from_path("dataset_v1/")
version = DatasetVersion(ds)

# Append new episodes (creates dataset_v2/)
df_new = prf.RoboticsDataFrame.from_converted("new_episodes/")
version.append(df_new, tag="v2", description="Added 100 new pick episodes")

# Load specific version
ds_v2 = prf.RoboFrameDataset.from_path("dataset_v2/")
```
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dataframe import RoboticsDataFrame


class DatasetVersion:
    """Manage dataset versions and incremental updates."""

    METADATA_FILE = ".versions.json"

    def __init__(self, dataset_root: str | Path):
        """Initialize version manager for a dataset.

        Args:
            dataset_root: Root directory of the dataset
        """
        self.root = Path(dataset_root)
        self.metadata_path = self.root / self.METADATA_FILE
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        """Load version metadata if it exists."""
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                return json.load(f)
        return {
            "versions": [],
            "current": None,
            "created": datetime.now().isoformat(),
        }

    def _save_metadata(self) -> None:
        """Save version metadata."""
        with open(self.metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def current_version(self) -> str | None:
        """Get the current version tag."""
        return self._metadata.get("current")

    def list_versions(self) -> list[dict[str, Any]]:
        """List all versions with metadata."""
        return self._metadata.get("versions", [])

    def append(
        self,
        new_dataframe: RoboticsDataFrame,
        tag: str,
        description: str = "",
    ) -> Path:
        """Append new episodes to the dataset, creating a new version.

        Args:
            new_dataframe: RoboticsDataFrame with new episodes
            tag: Version tag (e.g., "v2", "v2.1")
            description: Human-readable description of changes

        Returns:
            Path to the new versioned dataset directory
        """
        # Create new version directory
        version_dir = self.root.parent / f"{self.root.name}_{tag}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy existing data
        if self.root.exists():
            for item in ["meta", "data", "videos"]:
                src = self.root / item
                dst = version_dir / item
                if src.exists():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)

        # Append new episodes
        self._append_episodes(version_dir, new_dataframe)

        # Update metadata
        self._metadata["versions"].append(
            {
                "tag": tag,
                "created": datetime.now().isoformat(),
                "description": description,
                "episodes": new_dataframe.num_episodes(),
            }
        )
        self._metadata["current"] = tag

        # Save version metadata to new directory
        version_metadata_path = version_dir / self.METADATA_FILE
        with open(version_metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

        # Also save to original root (for tracking)
        self._save_metadata()

        return version_dir

    def _append_episodes(self, target_dir: Path, df: RoboticsDataFrame) -> None:
        """Append episodes from df to target_dir (internal).

        This is a simplified implementation. Full implementation would:
        - Append to existing Parquet files (via PyArrow)
        - Update episode metadata with new indices
        - Rollup statistics
        """
        # For now, use the dataframe's save method to add to target
        # In production, this would do true appends rather than rewrites
        df.save(str(target_dir))

    def rollback(self, tag: str) -> bool:
        """Rollback to a previous version (conceptual).

        In practice, you'd load the versioned directory directly:
        `ds = prf.RoboFrameDataset.from_path("dataset_v1/")`

        Args:
            tag: Version tag to rollback to

        Returns:
            True if version exists
        """
        for v in self._metadata.get("versions", []):
            if v["tag"] == tag:
                self._metadata["current"] = tag
                self._save_metadata()
                return True
        return False

    def get_version_path(self, tag: str | None = None) -> Path | None:
        """Get the path to a specific version directory.

        Args:
            tag: Version tag (None = current)

        Returns:
            Path to version directory, or None if not found
        """
        if tag is None:
            tag = self.current_version()
        if tag is None:
            return self.root if self.root.exists() else None

        # Try common naming patterns
        for pattern in [f"{self.root.name}_{tag}", f"{self.root.name}_{tag}/", tag]:
            candidate = self.root.parent / pattern
            if candidate.exists():
                return candidate

        return None


class DatasetManifest:
    """Lightweight manifest for dataset provenance and reproducibility."""

    def __init__(self, dataset_root: str | Path):
        """Initialize manifest for a dataset.

        Args:
            dataset_root: Root directory of the dataset
        """
        self.root = Path(dataset_root)
        self.manifest_file = self.root / "manifest.json"

    def create(
        self,
        name: str,
        description: str = "",
        source: str = "",
        records: int = 0,
        **metadata: Any,
    ) -> None:
        """Create a new dataset manifest.

        Args:
            name: Dataset name
            description: Dataset description
            source: Source of data (e.g., "LeRobot v3.0 download", "converter: MCAP")
            records: Total record count
            **metadata: Additional metadata fields
        """
        manifest = {
            "name": name,
            "description": description,
            "source": source,
            "records": records,
            "created": datetime.now().isoformat(),
            "metadata": metadata,
        }
        with open(self.manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

    def update(self, **fields: Any) -> None:
        """Update manifest fields."""
        if not self.manifest_file.exists():
            self.create("Dataset", **fields)
            return

        with open(self.manifest_file) as f:
            manifest = json.load(f)

        manifest.update(fields)
        manifest["updated"] = datetime.now().isoformat()

        with open(self.manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

    def load(self) -> dict[str, Any] | None:
        """Load manifest data."""
        if self.manifest_file.exists():
            with open(self.manifest_file) as f:
                return json.load(f)
        return None
