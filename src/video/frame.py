from dataclasses import dataclass
import numpy as np


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    image: np.ndarray