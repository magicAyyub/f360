from __future__ import annotations

from typing import Any, List

import numpy as np
import supervision as sv

from .base import Track, Tracker


def _to_detections(prediction: Any) -> sv.Detections:
    if isinstance(prediction, sv.Detections):
        return prediction

    if hasattr(prediction, "boxes") and hasattr(prediction.boxes, "xyxy"):
        return sv.Detections.from_ultralytics(prediction)

    as_np = lambda x: x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    if hasattr(prediction, "boxes") and hasattr(prediction, "scores"):
        boxes = as_np(prediction.boxes).reshape(-1, 4)
        scores = as_np(prediction.scores).ravel()
        cls = getattr(prediction, "class_ids", getattr(prediction, "classes", None))
        class_ids = as_np(cls).ravel().astype(int) if cls is not None else np.zeros(len(boxes), int)
        return sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)

    if isinstance(prediction, dict):
        boxes = as_np(prediction["boxes"]).reshape(-1, 4)
        scores = as_np(prediction.get("scores", np.ones(len(boxes)))).ravel()
        cls = prediction.get("class_ids", prediction.get("labels"))
        class_ids = as_np(cls).ravel().astype(int) if cls is not None else np.zeros(len(boxes), int)
        return sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)

    if isinstance(prediction, (tuple, list)) and len(prediction) >= 2:
        boxes = as_np(prediction[0]).reshape(-1, 4)
        scores = as_np(prediction[1]).ravel()
        class_ids = (
            as_np(prediction[2]).ravel().astype(int)
            if len(prediction) > 2 and prediction[2] is not None
            else np.zeros(len(boxes), int)
        )
        return sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)

    arr = as_np(prediction)
    if arr.ndim == 2 and arr.shape[1] >= 5:
        return sv.Detections(
            xyxy=arr[:, :4],
            confidence=arr[:, 4],
            class_id=arr[:, 5].astype(int) if arr.shape[1] >= 6 else np.zeros(len(arr), int),
        )

    raise TypeError(f"Unsupported prediction format: {type(prediction)!r}")


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