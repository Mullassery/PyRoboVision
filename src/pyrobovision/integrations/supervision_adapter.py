"""Optional Supervision adapter.

Supervision (MIT) is the common detection/tracking utility layer used with
Ultralytics, MMDetection, YOLO-NAS, etc. This adapter converts its
``Detections`` objects into PyRoboVision detections and back, so users can
bring any detector without PyRoboVision taking an AGPL dependency.

Ultralytics itself is **not** a dependency or supported extra here: it is
AGPL-3.0 and cannot be bundled with this Apache-2.0 package.

Install::

    pip install "pyrobovision[supervision]"
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..tracking.mot import Detection, MOTTracker, Track


class SupervisionUnavailable(RuntimeError):
    """Raised when a Supervision-specific conversion is requested without it."""


def _require_supervision():
    try:
        import supervision as sv
    except ImportError as exc:
        raise SupervisionUnavailable(
            "Supervision is not installed. Install the optional adapter with "
            '`pip install "pyrobovision[supervision]"` (or `pip install supervision`).'
        ) from exc
    return sv


def _as_array(value: Any, *, dtype=float) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def detections_from_supervision(detections: Any) -> list[Detection]:
    """Convert ``sv.Detections`` (or a compatible object) to PyRoboVision.

    Only the documented public fields are read: ``xyxy``, ``confidence``,
    ``class_id``, and optional ``data``/``tracker_id``. Features are taken
    from ``data["features"]`` when present.
    """
    xyxy = _as_array(getattr(detections, "xyxy", None))
    if xyxy is None:
        raise ValueError("detections must expose an xyxy array")
    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ValueError(f"xyxy must have shape (N, 4), got {xyxy.shape}")

    confidence = _as_array(getattr(detections, "confidence", None))
    class_id = _as_array(getattr(detections, "class_id", None), dtype=int)
    data = getattr(detections, "data", None) or {}
    features = data.get("features") if isinstance(data, dict) else None

    result: list[Detection] = []
    for index, box in enumerate(xyxy):
        result.append(
            Detection(
                bbox=box.astype(float),
                confidence=float(confidence[index]) if confidence is not None and index < len(confidence) else 1.0,
                class_id=int(class_id[index]) if class_id is not None and index < len(class_id) else 0,
                features=np.asarray(features[index]) if features is not None else None,
            )
        )
    return result


def tracks_to_supervision(
    tracks: Iterable[Track],
    *,
    class_id: int = 0,
    existing: Any = None,
) -> Any:
    """Convert tracked boxes to ``sv.Detections`` with ``tracker_id`` set."""
    sv = _require_supervision()
    tracks = list(tracks)
    if not tracks:
        if existing is not None:
            return existing
        return sv.Detections.empty()

    xyxy = np.stack([track.get_bbox().astype(float) for track in tracks])
    confidence = np.array(
        [
            track.detections[-1][1].confidence if track.detections else 1.0
            for track in tracks
        ],
        dtype=float,
    )
    tracker_id = np.array([track.track_id for track in tracks], dtype=int)
    class_ids = np.array(
        [
            track.detections[-1][1].class_id if track.detections else class_id
            for track in tracks
        ],
        dtype=int,
    )
    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_ids,
        tracker_id=tracker_id,
    )


class SupervisionTrackerAdapter:
    """Run PyRoboVision's MOTTracker on ``sv.Detections``.

    Example::

        adapter = SupervisionTrackerAdapter()
        tracked = adapter.update(sv_detections)  # sv.Detections with tracker_id
    """

    def __init__(self, tracker: MOTTracker | None = None):
        self.tracker = tracker or MOTTracker()

    def update(self, detections: Any) -> Any:
        converted = detections_from_supervision(detections)
        tracks = self.tracker.update(converted)
        return tracks_to_supervision(tracks, existing=detections)

    def reset(self) -> None:
        self.tracker.reset()


__all__ = [
    "SupervisionTrackerAdapter",
    "SupervisionUnavailable",
    "detections_from_supervision",
    "tracks_to_supervision",
]
