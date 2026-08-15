import cv2
import numpy as np
from typing import Iterator, Optional, Tuple
from dataclasses import dataclass

Size = Tuple[int, int]

DEFAULT_FPS = 30.0


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    image: np.ndarray


class VideoReader:
    """Iterable wrapper around a video file.

    Yields frames within an optional time window, keeping one frame every
    `stride` frames and optionally resizing them to `resize`.
    """

    def __init__(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        stride: int = 1,
        resize: Optional[Size] = None,
    ):
        if start_time < 0:
            raise ValueError("start_time must be non-negative")
        if end_time is not None and end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        if stride < 1:
            raise ValueError("stride must be a positive integer")

        self.video_path = video_path
        self.start_time = start_time
        self.end_time = end_time
        self.stride = stride
        self.resize = resize

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"cannot open video file: {self.video_path}")

        try:
            fps = _read_fps(cap)
            start_frame = int(self.start_time * fps)
            end_frame = int(self.end_time * fps) if self.end_time is not None else None

            # On saute directement au début de la fenêtre au lieu de décoder
            # puis jeter tout ce qui précède.
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_id = start_frame
            while end_frame is None or frame_id <= end_frame:
                # grab() avance sans décoder: seules les frames retenues
                # par le stride paient le coût du décodage.
                if not cap.grab():
                    break

                if (frame_id - start_frame) % self.stride == 0:
                    success, image = cap.retrieve()
                    if not success:
                        break

                    yield Frame(
                        frame_id=frame_id,
                        timestamp=frame_id / fps,
                        image=self._resize(image) if self.resize else image,
                    )

                frame_id += 1
        finally:
            cap.release()

    @property
    def fps(self) -> float:
        return self._probe()[0]

    @property
    def frame_count(self) -> int:
        """How many frames this reader will yield, useful to size a progress bar."""
        fps, total = self._probe()

        first = int(self.start_time * fps)
        last = total - 1
        if self.end_time is not None:
            last = min(int(self.end_time * fps), last)

        return max(0, (last - first) // self.stride + 1)

    def _probe(self) -> Tuple[float, int]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"cannot open video file: {self.video_path}")

        try:
            return _read_fps(cap), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    def _resize(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.resize, interpolation=cv2.INTER_AREA)


def _read_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps > 0 else DEFAULT_FPS
