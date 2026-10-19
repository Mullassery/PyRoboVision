"""Supervision adapter tests use a duck-typed Detections object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pyrobovision.integrations.supervision_adapter import (
    SupervisionTrackerAdapter,
    detections_from_supervision,
    tracks_to_supervision,
)


@dataclass
class _FakeDetections:
    xyxy: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    data: dict | None = None


def test_conversion_preserves_boxes_confidence_and_class():
    detections = _FakeDetections(
        xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([1, 2]),
    )
    converted = detections_from_supervision(detections)
    assert len(converted) == 2
    assert converted[0].confidence == 0.9
    assert converted[1].class_id == 2
    np.testing.assert_allclose(converted[0].bbox, [0, 0, 10, 10])


def test_tracker_adapter_runs_without_supervision_installed():
    try:
        import supervision  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("supervision is installed; covered by real round-trip test")
    tracker = SupervisionTrackerAdapter()
    detections = _FakeDetections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([1]),
    )
    with pytest.raises(Exception, match="Supervision is not installed"):
        tracker.update(detections)


def test_rejects_bad_shapes():
    import pytest

    bad = _FakeDetections(
        xyxy=np.array([0, 0, 10, 10]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    with pytest.raises(ValueError, match="shape"):
        detections_from_supervision(bad)


def test_real_supervision_detections_roundtrip():
    sv = pytest.importorskip("supervision")
    import numpy as np

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([1]),
    )
    converted = detections_from_supervision(detections)
    assert len(converted) == 1
    assert converted[0].confidence == 0.9
    out = tracks_to_supervision([])
    assert out is not None
