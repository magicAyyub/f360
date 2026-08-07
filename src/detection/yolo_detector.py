from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np
from ultralytics import YOLO

from .base import Detector, Prediction


class YoloDetector(Detector):
    def __init__(
        self,
        model_path: str = 'models/yolov5su.pt',
        conf: float = 0.25,
        classes: Optional[Sequence[int]] = None,
    ):
        # Fallback to yolov5s.pt if the default model is missing
        if model_path == 'models/yolov5su.pt' and not os.path.exists(model_path):
            model_path = 'models/yolov5s.pt'

        # Ensure the models directory exists so downloads do not clutter the root
        parent_dir = os.path.dirname(model_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = list(classes) if classes is not None else None

    def predict(self, frame: Any) -> Prediction:
        result = self.model.predict(
            source=frame,
            conf=self.conf,
            classes=self.classes,
            verbose=False,
        )
        boxes = result[0].boxes

        if boxes is None or len(boxes) == 0:
            return Prediction(
                boxes=np.zeros((0, 4), dtype=float),
                scores=np.zeros((0,), dtype=float),
                classes=np.zeros((0,), dtype=int),
            )

        # .cpu() est nécessaire: les tenseurs restent sur MPS/CUDA sinon
        return Prediction(
            boxes=boxes.xyxy.cpu().numpy(),
            scores=boxes.conf.cpu().numpy(),
            classes=boxes.cls.cpu().numpy().astype(int),
        )
