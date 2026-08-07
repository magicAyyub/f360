from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Prediction:
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray


class Detector(ABC):
    @abstractmethod
    def predict(self, frame: Any) -> Prediction:
        raise NotImplementedError
