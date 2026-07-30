import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from .kalman_filter import KalmanFilter
from .association import ObjectAssociation


@dataclass
class Detection:
    bbox: np.ndarray
    confidence: float
    features: Optional[np.ndarray] = None
    class_id: int = 0


@dataclass
class Track:
    track_id: int
    detections: List[Tuple[int, Detection]] = field(default_factory=list)
    kalman_filter: KalmanFilter = field(default_factory=KalmanFilter)
    age: int = 1
    time_since_update: int = 0
    hit_streak: int = 1
    start_frame: int = 0
    is_confirmed: bool = False
    state_history: List[np.ndarray] = field(default_factory=list)

    def update(self, detection: Detection, frame_id: int) -> None:
        self.detections.append((frame_id, detection))
        self.time_since_update = 0
        self.hit_streak += 1
        self.age += 1
        if self.hit_streak >= 3:
            self.is_confirmed = True

        z = np.array([(detection.bbox[0] + detection.bbox[2]) / 2,
                      (detection.bbox[1] + detection.bbox[3]) / 2])
        self.kalman_filter.update(z)
        self.state_history.append(self.kalman_filter.get_state()[0].copy())

    def predict(self, dt: float = 1.0) -> np.ndarray:
        self.time_since_update += 1
        state = self.kalman_filter.predict(dt)
        self.state_history.append(state.copy())
        return state

    def get_position(self) -> np.ndarray:
        return self.kalman_filter.x[0:2].flatten()

    def get_velocity(self) -> np.ndarray:
        return self.kalman_filter.x[2:4].flatten()

    def get_bbox(self) -> np.ndarray:
        if len(self.detections) == 0:
            x, y = self.get_position()
            w, h = 50, 50
            return np.array([x - w/2, y - h/2, x + w/2, y + h/2])
        frame_id, det = self.detections[-1]
        return det.bbox

    def is_alive(self, max_age: int = 30) -> bool:
        return self.time_since_update < max_age


class MOTTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.association = ObjectAssociation(iou_threshold=iou_threshold)
        self.tracks: List[Track] = []
        self.frame_id = 0
        self.next_track_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        self.frame_id += 1

        for track in self.tracks:
            track.predict()

        track_boxes = [track.get_bbox() for track in self.tracks]
        detection_boxes = [det.bbox for det in detections]

        matches, unmatched_tracks, unmatched_detections = self.association.associate(
            track_boxes, detection_boxes
        )

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], self.frame_id)

        for det_idx in unmatched_detections:
            self._create_track(detections[det_idx])

        self.tracks = [t for t in self.tracks if t.is_alive(self.max_age)]

        return self.get_confirmed_tracks()

    def _create_track(self, detection: Detection) -> None:
        track = Track(
            track_id=self.next_track_id,
            start_frame=self.frame_id,
        )
        self.next_track_id += 1

        pos = np.array([(detection.bbox[0] + detection.bbox[2]) / 2,
                        (detection.bbox[1] + detection.bbox[3]) / 2])
        track.kalman_filter.initialize(pos)
        track.update(detection, self.frame_id)
        self.tracks.append(track)

    def get_confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.is_confirmed]

    def get_all_tracks(self) -> List[Track]:
        return self.tracks.copy()

    def reset(self) -> None:
        self.tracks = []
        self.frame_id = 0
        self.next_track_id = 1

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_statistics(self) -> Dict[str, int]:
        return {
            "total_tracks": len(self.tracks),
            "confirmed_tracks": len(self.get_confirmed_tracks()),
            "current_frame": self.frame_id,
            "next_track_id": self.next_track_id,
        }
