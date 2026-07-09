import numpy as np
import pytest
from src.analytics import PlayPhaseDetector
from src.detection import Detector, Prediction


class DummyDetector(Detector):
    def __init__(self, boxes, classes):
        self.boxes = np.asarray(boxes)
        self.classes = np.asarray(classes)

    def predict(self, frame) -> Prediction:
        scores = np.ones(len(self.boxes))
        return Prediction(boxes=self.boxes, scores=scores, classes=self.classes)


def test_green_ratio():
    detector = PlayPhaseDetector()
    
    # Fully black image (no green)
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.compute_green_ratio(black_img) == 0.0

    # Fully green image (Hue 60, Sat 255, Val 255 matches [0, 255, 0] BGR)
    green_img = np.zeros((100, 100, 3), dtype=np.uint8)
    green_img[:, :] = [0, 255, 0]
    assert detector.compute_green_ratio(green_img) == 1.0


def test_is_in_play_low_green():
    detector = PlayPhaseDetector(min_green_ratio=0.5)
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    is_play, metrics = detector.is_in_play(black_img)
    assert not is_play
    assert metrics["reason"] == "low_green_ratio"


def test_is_in_play_no_detector():
    detector = PlayPhaseDetector(min_green_ratio=0.5)
    green_img = np.zeros((100, 100, 3), dtype=np.uint8)
    green_img[:, :] = [0, 255, 0]

    is_play, metrics = detector.is_in_play(green_img)
    assert is_play
    assert metrics["reason"] == "high_green_ratio_no_detector"


def test_is_in_play_insufficient_players():
    detector = PlayPhaseDetector(min_green_ratio=0.5, min_players=3)
    green_img = np.zeros((100, 100, 3), dtype=np.uint8)
    green_img[:, :] = [0, 255, 0]

    # Detector returns 2 people (less than min_players=3)
    dummy_det = DummyDetector(
        boxes=[[10, 10, 20, 20], [30, 30, 40, 40]],
        classes=[0, 0]
    )

    is_play, metrics = detector.is_in_play(green_img, dummy_det)
    assert not is_play
    assert metrics["reason"] == "insufficient_players"
    assert metrics["num_players"] == 2


def test_is_in_play_close_up():
    detector = PlayPhaseDetector(min_green_ratio=0.5, min_players=2, max_player_height_ratio=0.2)
    green_img = np.zeros((100, 100, 3), dtype=np.uint8)
    green_img[:, :] = [0, 255, 0]

    # Frame height is 100.
    # Player 1 height: 30 (ratio = 0.3 > 0.2) -> Close up!
    # Player 2 height: 10 (ratio = 0.1)
    dummy_det = DummyDetector(
        boxes=[[10, 10, 20, 40], [30, 30, 40, 40]],
        classes=[0, 0]
    )

    is_play, metrics = detector.is_in_play(green_img, dummy_det)
    assert not is_play
    assert metrics["reason"] == "close_up"
    assert metrics["max_player_height_ratio"] == 0.3


def test_is_in_play_success():
    detector = PlayPhaseDetector(min_green_ratio=0.5, min_players=2, max_player_height_ratio=0.2)
    green_img = np.zeros((100, 100, 3), dtype=np.uint8)
    green_img[:, :] = [0, 255, 0]

    # Both players have height ratio 0.1 <= 0.2
    dummy_det = DummyDetector(
        boxes=[[10, 10, 20, 20], [30, 30, 40, 40]],
        classes=[0, 0]
    )

    is_play, metrics = detector.is_in_play(green_img, dummy_det)
    assert is_play
    assert metrics["reason"] == "in_play"
    assert metrics["num_players"] == 2
