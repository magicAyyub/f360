from __future__ import annotations

from typing import Any, List

import numpy as np
import supervision as sv

from .base import Track, Tracker


def _to_detections(prediction: Any) -> sv.Detections:
    if isinstance(prediction, sv.Detections):
        return prediction

    return sv.Detections(
        xyxy=np.asarray(prediction.boxes, dtype=float).reshape(-1, 4),
        confidence=np.asarray(prediction.scores, dtype=float).ravel(),
        class_id=np.asarray(prediction.classes, dtype=int).ravel(),
    )


class ByteTrackTracker(Tracker):

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: float = 30.0,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    def reset(self) -> None:
        self._tracker.reset()

    def update(self, prediction: Any) -> List[Track]:
        tracked = self._tracker.update_with_detections(_to_detections(prediction))
        return [
            Track(
                track_id=int(tid),
                box=xyxy,
                score=float(conf),
                class_id=int(cls),
            )
            for xyxy, conf, cls, tid in zip(
                tracked.xyxy,
                tracked.confidence if tracked.confidence is not None else np.zeros(len(tracked)),
                tracked.class_id if tracked.class_id is not None else np.zeros(len(tracked), int),
                tracked.tracker_id if tracked.tracker_id is not None else range(len(tracked)),
            )
        ]