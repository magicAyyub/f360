import cv2
import numpy as np
from typing import Iterable, Iterator, Optional, Tuple
from dataclasses import dataclass

Size = Tuple[int, int]

@dataclass
class Frame:
    frame_id: int
    timestamp: float
    image: np.ndarray

class VideoReader:
    """Iterable wrapper around a video file, yielding frames with timestamps."""

    def __init__(self, video_path: str):
        self.video_path = video_path

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frame_id = 0
        while True:
            success, image = cap.read()
            if not success:
                break

            yield Frame(
                frame_id=frame_id,
                timestamp=frame_id / fps,
                image=image,
            )
            frame_id += 1

        cap.release()


class FrameSampler:
    """Sample frames from a frame iterable using stride and optional resize."""

    def __init__(self, source: Iterable[Frame], stride: int = 1, resize: Optional[Size] = None):
        if stride < 1:
            raise ValueError("stride must be a positive integer")

        self.source = source
        self.stride = stride
        self.resize = resize

    def __iter__(self) -> Iterator[Frame]:
        for index, frame in enumerate(self.source):
            if index % self.stride != 0:
                continue

            image = self._resize(frame.image) if self.resize else frame.image
            yield Frame(frame_id=frame.frame_id, timestamp=frame.timestamp, image=image)

    def _resize(self, image):
        return cv2.resize(image, self.resize, interpolation=cv2.INTER_AREA)


class TimeWindowFilter:
    """Filter frames by a start/end timestamp window."""

    def __init__(
        self,
        source: Iterable[Frame],
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ):
        if start_time < 0:
            raise ValueError("start_time must be non-negative")
        if end_time is not None and end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")

        self.source = source
        self.start_time = start_time
        self.end_time = end_time

    def __iter__(self) -> Iterator[Frame]:
        for frame in self.source:
            if frame.timestamp < self.start_time:
                continue
            if self.end_time is not None and frame.timestamp > self.end_time:
                break
            yield frame
