from __future__ import annotations

import os
from typing import List
import numpy as np
from ultralytics import SAM


class SamSegmenter:
    """Segment Anything Model 2 wrapper to generate pixel-accurate masks from bounding boxes."""

    def __init__(self, model_path: str = 'models/sam2_t.pt') -> None:
        # Fallback to default if model file is missing
        if model_path == 'models/sam2_t.pt' and not os.path.exists(model_path):
            parent_dir = os.path.dirname(model_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        self.model = SAM(model_path)

    def segment(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        """
        Generate binary masks for a list of bounding boxes on the given frame.

        Args:
            frame: BGR image frame.
            bboxes: Bounding boxes of shape (N, 4) in XYXY format.

        Returns:
            A list of boolean numpy arrays of shape (H, W) corresponding to the mask of each box.
        """
        if len(bboxes) == 0:
            return []

        # Predict masks using SAM2
        results = self.model.predict(frame, bboxes=bboxes, verbose=False)
        
        if not results or results[0].masks is None:
            return [np.zeros(frame.shape[:2], dtype=bool) for _ in bboxes]

        # Get binary masks from prediction
        masks_data = results[0].masks.data.cpu().numpy()
        
        # Verify length matches bboxes
        masks = []
        for i in range(len(bboxes)):
            if i < len(masks_data):
                masks.append(masks_data[i] > 0.5)
            else:
                masks.append(np.zeros(frame.shape[:2], dtype=bool))
                
        return masks
