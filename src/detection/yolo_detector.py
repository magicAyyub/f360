from __future__ import annotations

import os
from typing import Any

import numpy as np
from ultralytics import YOLO

from .base import Detector, Prediction


class YoloDetector(Detector):
    def __init__(self, model_path: str = 'models/yolov5su.pt', conf: float = 0.25):
        # Fallback to yolov5s.pt if the default model is missing
        if model_path == 'models/yolov5su.pt' and not os.path.exists(model_path):
            model_path = 'models/yolov5s.pt'

        # Ensure the models directory exists so downloads do not clutter the root
        parent_dir = os.path.dirname(model_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame: Any) -> Prediction:
        result = self.model.predict(source=frame, conf=self.conf, verbose=False)
        prediction = result[0]

        boxes = prediction.boxes.xyxy.numpy() if prediction.boxes is not None else np.zeros((0, 4), dtype=float)
        scores = prediction.boxes.conf.numpy() if prediction.boxes is not None else np.zeros((0,), dtype=float)
        classes = prediction.boxes.cls.numpy().astype(int) if prediction.boxes is not None else np.zeros((0,), dtype=int)

        return Prediction(boxes=boxes, scores=scores, classes=classes)

