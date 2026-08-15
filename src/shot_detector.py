import cv2
import json
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from src.reader import Frame
from src.vendor.transnetv2_pytorch import TransNetV2

DEFAULT_WEIGHTS = "models/transnetv2/transnetv2-pytorch-weights.pth"

MODEL_INPUT_SIZE = (48, 27)
WINDOW_SIZE = 100
# Chaque fenêtre garde 25 frames de contexte de chaque côté: seules les 50
# prédictions centrales sont conservées.
WINDOW_CONTEXT = 25
WINDOW_STEP = WINDOW_SIZE - 2 * WINDOW_CONTEXT


def _best_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Shot:
    """A continuous camera shot, bounded by two transitions.

    Frame bounds are inclusive. `end_time` is exclusive, so it is the exact cut
    point: a segment can be extracted with `-ss start_time -to end_time` without
    losing or duplicating a frame.
    """

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class ShotDetector:
    """Shot boundary detection with a pre-trained TransNetV2.

    The model looks at 100 consecutive frames downscaled to 48x27 and outputs,
    for each frame, the probability of being part of a transition. Frames must
    therefore be consecutive: reading with a stride would fabricate cuts.
    """

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS, threshold: float = 0.5, device: Optional[str] = None):
        self.device = torch.device(device or _best_available_device())
        self.threshold = threshold

        self.model = TransNetV2()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval().to(self.device)

    def detect(self, frames: Iterable[Frame]) -> List[Shot]:
        probabilities, timeline = self.predict(frames)
        return self.to_shots(probabilities, timeline)

    def predict(self, frames: Iterable[Frame]) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
        """Transition probability per frame, computed as the frames arrive.

        Only one window is ever held in memory, so a full match costs no more
        than a one minute extract.
        """
        window, timeline, probabilities = [], [], []
        emitted = 0

        for frame in frames:
            image = self._prepare(frame.image)
            if not timeline:
                # Contexte gauche de la premiere fenetre: la frame initiale repetee.
                window.extend([image] * WINDOW_CONTEXT)

            window.append(image)
            timeline.append((frame.frame_id, frame.timestamp))

            if len(window) == WINDOW_SIZE:
                probabilities.append(self._run(window))
                del window[:WINDOW_STEP]
                emitted += WINDOW_STEP

        if not timeline:
            raise ValueError("no frame to analyse")

        # Contexte droit, meme principe, jusqu'a couvrir les frames encore en attente.
        while emitted < len(timeline):
            window.extend([window[-1]] * (WINDOW_SIZE - len(window)))
            probabilities.append(self._run(window))
            del window[:WINDOW_STEP]
            emitted += WINDOW_STEP

        return np.concatenate(probabilities)[:len(timeline)], timeline

    def _run(self, window: List[np.ndarray]) -> np.ndarray:
        """Probabilities for the 50 frames sitting at the centre of one window."""
        batch = torch.from_numpy(np.stack(window)).unsqueeze(0)
        with torch.no_grad():
            single_frame_pred, _ = self.model(batch.to(self.device))
            return torch.sigmoid(single_frame_pred)[0, WINDOW_CONTEXT:-WINDOW_CONTEXT, 0].cpu().numpy()

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        small = cv2.resize(image, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def to_shots(self, probabilities: np.ndarray, timeline: List[Tuple[int, float]]) -> List[Shot]:
        """Turn per-frame probabilities into the shots between the transitions."""
        in_transition = probabilities > self.threshold

        bounds, start, previous = [], 0, False
        for index, transition in enumerate(in_transition):
            if previous and not transition:
                start = index
            if transition and not previous and index != 0:
                # index est la premiere frame de transition: elle melange les deux
                # plans, on la laisse donc en dehors du plan sortant.
                bounds.append((start, index - 1))
            previous = transition

        if not in_transition[-1]:
            bounds.append((start, len(in_transition) - 1))
        if not bounds:
            bounds = [(0, len(in_transition) - 1)]

        frame_duration = timeline[1][1] - timeline[0][1] if len(timeline) > 1 else 0.0

        return [
            Shot(
                start_frame=timeline[first][0],
                end_frame=timeline[last][0],
                start_time=timeline[first][1],
                end_time=timeline[last][1] + frame_duration,
            )
            for first, last in bounds
        ]


TIME_PRECISION = 3


def save_shots(shots: Sequence[Shot], path: str, video_path: Optional[str] = None) -> None:
    """Write the shot list as JSON, timestamps rounded to the millisecond."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "video": video_path,
        "shot_count": len(shots),
        "shots": [_as_entry(index, shot) for index, shot in enumerate(shots, start=1)],
    }
    destination.write_text(json.dumps(payload, indent=2))


def _as_entry(index: int, shot: Shot) -> dict:
    return {
        "index": index,
        "start_frame": shot.start_frame,
        "end_frame": shot.end_frame,
        "start_time": round(shot.start_time, TIME_PRECISION),
        "end_time": round(shot.end_time, TIME_PRECISION),
        "duration": round(shot.duration, TIME_PRECISION),
    }
