from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

import numpy as np


@dataclass
class Track:
    track_id: int
    box: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_id: int


class Tracker(ABC):
    @abstractmethod
    def update(self, prediction: Any) -> List[Track]:
        raise NotImplementedError
