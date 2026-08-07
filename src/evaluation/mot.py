from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from src.tracking.base import Track


# frame, track_id, x, y, w, h, conf
COLUMNS = ('frame', 'track_id', 'x', 'y', 'w', 'h', 'conf')

# Le format MOTChallenge numérote les frames à partir de 1
FIRST_FRAME = 1


def xyxy_to_tlwh(boxes: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] boxes to MOT [left, top, width, height]."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack([
        boxes[:, 0],
        boxes[:, 1],
        boxes[:, 2] - boxes[:, 0],
        boxes[:, 3] - boxes[:, 1],
    ])


def tracks_to_mot(frame: int, tracks: Iterable[Track]) -> np.ndarray:
    """Convert one frame of tracker output to MOT rows.

    frame is the 1-based position within the evaluated sequence, which is not
    Frame.frame_id: that one indexes the source video and survives seeking.
    """
    tracks = list(tracks)
    if not tracks:
        return np.zeros((0, len(COLUMNS)), dtype=float)

    tlwh = xyxy_to_tlwh(np.array([t.box for t in tracks]))
    return np.column_stack([
        np.full(len(tracks), frame, dtype=float),
        np.array([t.track_id for t in tracks], dtype=float),
        tlwh,
        np.array([t.score for t in tracks], dtype=float),
    ])


def load_mot(path: str | Path) -> np.ndarray:
    """Read a MOTChallenge annotation file into an (N, 7) array.

    Rows flagged as inactive or zero-confidence are dropped, which is how the
    MOT ground truth marks regions to ignore.
    """
    rows = np.loadtxt(str(path), delimiter=',', ndmin=2)
    if rows.size == 0:
        return np.zeros((0, len(COLUMNS)), dtype=float)

    parsed = rows[:, :len(COLUMNS)]
    return parsed[parsed[:, 6] > 0]


def save_mot(rows: np.ndarray, path: str | Path) -> None:
    rows = np.asarray(rows, dtype=float).reshape(-1, len(COLUMNS))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), rows, delimiter=',', fmt='%.2f')


def group_by_frame(rows: np.ndarray, frames: Sequence[int]) -> Dict[int, np.ndarray]:
    """Index rows by frame number, with an empty entry for frames without objects."""
    rows = np.asarray(rows, dtype=float).reshape(-1, len(COLUMNS))
    grouped = {int(f): np.zeros((0, len(COLUMNS)), dtype=float) for f in frames}

    for frame in np.unique(rows[:, 0]).astype(int):
        if frame in grouped:
            grouped[frame] = rows[rows[:, 0] == frame]

    return grouped


def frame_range(*row_sets: np.ndarray) -> np.ndarray:
    """Every frame number covered by the given row sets, sorted."""
    frames = [np.asarray(r, dtype=float).reshape(-1, len(COLUMNS))[:, 0] for r in row_sets]
    stacked = np.concatenate(frames) if frames else np.zeros(0)
    if stacked.size == 0:
        return np.zeros(0, dtype=int)
    return np.arange(int(stacked.min()), int(stacked.max()) + 1)
