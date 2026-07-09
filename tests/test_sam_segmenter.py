import sys
import types
import numpy as np
import pytest


def test_sam_segmenter(monkeypatch):
    class DummyMasks:
        def __init__(self, data):
            self.data = data

    class DummyResult:
        def __init__(self, masks_data):
            import torch
            self.masks = DummyMasks(torch.as_tensor(masks_data))

    class DummySAM:
        def __init__(self, model_path):
            self.model_path = model_path

        def predict(self, frame, bboxes, verbose=False):
            h, w = frame.shape[:2]
            n = len(bboxes)
            masks_data = np.zeros((n, h, w), dtype=bool)
            for i in range(n):
                masks_data[i, 10:15, 10:15] = True
            return [DummyResult(masks_data)]

    # Mock the SAM class in ultralytics package
    import ultralytics
    monkeypatch.setattr(ultralytics, 'SAM', DummySAM)

    from src.tracking.sam_segmenter import SamSegmenter

    segmenter = SamSegmenter('models/sam2_t.pt')
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bboxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])

    masks = segmenter.segment(frame, bboxes)

    assert len(masks) == 2
    assert masks[0].shape == (100, 100)
    assert masks[0][12, 12] is True or masks[0][12, 12] == True
    assert not masks[0][0, 0]
