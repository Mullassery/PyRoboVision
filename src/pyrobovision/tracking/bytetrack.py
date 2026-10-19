"""ByteTrack-style two-stage association.

ByteTrack's key idea is to keep low-confidence detections for a second
association pass instead of discarding them. This implementation reuses the
existing Kalman/Hungarian tracker primitives and adds the high/low confidence
split, making it usable without an external AGPL/ultralytics dependency.
"""

from __future__ import annotations

from .mot import Detection, MOTTracker, Track


class ByteTrackTracker(MOTTracker):
    """Two-stage IoU association with high/low confidence detections."""

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        super().__init__(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def update(self, detections: list[Detection]) -> list[Track]:
        self.frame_id += 1
        for track in self.tracks:
            track.predict()

        high = [d for d in detections if d.confidence >= self.high_threshold]
        low = [
            d
            for d in detections
            if self.low_threshold <= d.confidence < self.high_threshold
        ]

        track_boxes = [track.get_bbox() for track in self.tracks]
        matches, unmatched_tracks, unmatched_high = self.association.associate(
            track_boxes, [d.bbox for d in high]
        )
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(high[det_idx], self.frame_id)

        # Second association: low-confidence detections may recover occluded
        # tracks, but never create a new track on their own.
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        remaining_boxes = [track.get_bbox() for track in remaining_tracks]
        low_matches, _low_unmatched_tracks, _ = self.association.associate(
            remaining_boxes, [d.bbox for d in low]
        )
        for local_idx, det_idx in low_matches:
            remaining_tracks[local_idx].update(low[det_idx], self.frame_id)

        # Only high-confidence unmatched detections start new tracks.
        for det_idx in unmatched_high:
            self._create_track(high[det_idx])

        self.tracks = [t for t in self.tracks if t.is_alive(self.max_age)]
        return self.get_confirmed_tracks()


__all__ = ["ByteTrackTracker"]
