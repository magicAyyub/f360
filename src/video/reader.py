import cv2
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

from .frame import Frame


Size = Tuple[int, int]


class VideoReader:
    """Iterable wrapper around a video file, yielding frames with timestamps.

    When start_time is set the capture seeks instead of decoding every preceding
    frame. The seek lands on the nearest keyframe, so it may start slightly before
    the requested time; wrap with TimeWindowFilter when the boundary must be exact.
    """

    def __init__(self, video_path: str, start_time: float = 0.0):
        if start_time < 0:
            raise ValueError("start_time must be non-negative")

        self.video_path = video_path
        self.start_time = start_time
        self._fps: Optional[float] = None

    @property
    def fps(self) -> float:
        if self._fps is None:
            cap = cv2.VideoCapture(self.video_path)
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        return self._fps

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._fps = fps

        frame_id = 0
        if self.start_time > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.start_time * fps))
            # On relit la position réelle: le seek atterrit sur une keyframe
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

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


class ImageSequenceReader:
    """Iterable over a folder of numbered frames, the layout MOT datasets ship.

    The frame rate is not discoverable from the files, so it comes from the
    caller, usually out of the dataset's own seqinfo.
    """

    def __init__(self, image_dir: str | Path, fps: float, extension: str = '.jpg'):
        if fps <= 0:
            raise ValueError("fps must be positive")

        self.image_dir = Path(image_dir)
        self._fps = fps
        self.paths = sorted(self.image_dir.glob(f'*{extension}'))

        if not self.paths:
            raise FileNotFoundError(f"no '{extension}' frames in {self.image_dir}")

    @property
    def fps(self) -> float:
        return self._fps

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Frame]:
        for index, path in enumerate(self.paths):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"could not read frame {path}")

            yield Frame(frame_id=index, timestamp=index / self._fps, image=image)


class FrameSampler:
    """Sample frames from a frame iterable using stride and optional resize."""

    def __init__(self, source: Iterable[Frame], stride: int = 1, resize: Optional[Size] = None):
        if stride < 1:
            raise ValueError("stride must be a positive integer")

        self.source = source
        self.stride = stride
        self.resize = resize

    @property
    def fps(self) -> float:
        """Rate at which this sampler emits frames, not the rate of the source."""
        return self.source.fps / self.stride

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

    @property
    def fps(self) -> float:
        return self.source.fps

    def __iter__(self) -> Iterator[Frame]:
        for frame in self.source:
            if frame.timestamp < self.start_time:
                continue
            if self.end_time is not None and frame.timestamp > self.end_time:
                break
            yield frame
