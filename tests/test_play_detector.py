import numpy as np
import pytest
from src.analytics import PlayPhaseDetector
from src.detection import Detector, Prediction


class DummyDetector(Detector):
    def __init__(self, boxes, classes):
        self.boxes = np.asarray(boxes)
        self.classes = np.asarray(classes)
        self.calls = 0

    def predict(self, frame) -> Prediction:
        self.calls += 1
        scores = np.ones(len(self.boxes))
        return Prediction(boxes=self.boxes, scores=scores, classes=self.classes)


def green_frame(size=100):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = [0, 255, 0]
    return frame


def test_green_ratio():
    detector = PlayPhaseDetector()

    # Fully black image (no green)
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.compute_green_ratio(black_img) == 0.0

    # Fully green image (Hue 60, Sat 255, Val 255 matches [0, 255, 0] BGR)
    assert detector.compute_green_ratio(green_frame()) == 1.0


def test_low_green_skips_detector():
    detector = PlayPhaseDetector(min_green_ratio=0.5)
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_det = DummyDetector(boxes=[[10, 10, 20, 20]], classes=[0])

    phase = detector.evaluate(black_img, dummy_det)
    assert not phase.in_play
    assert phase.metrics["reason"] == "low_green_ratio"
    # Le court-circuit doit épargner l'inférence
    assert dummy_det.calls == 0
    assert phase.prediction is None


def test_no_detector():
    detector = PlayPhaseDetector(min_green_ratio=0.5)

    phase = detector.evaluate(green_frame())
    assert phase.in_play
    assert phase.metrics["reason"] == "high_green_ratio_no_detector"
    assert phase.prediction is None


def test_insufficient_players():
    detector = PlayPhaseDetector(min_green_ratio=0.5, min_players=3)

    # Detector returns 2 people (less than min_players=3)
    dummy_det = DummyDetector(
        boxes=[[10, 10, 20, 20], [30, 30, 40, 40]],
        classes=[0, 0]
    )

    phase = detector.evaluate(green_frame(), dummy_det)
    assert not phase.in_play
    assert phase.metrics["reason"] == "insufficient_players"
    assert phase.metrics["num_players"] == 2


def test_close_up():
    detector = PlayPhaseDetector(min_green_ratio=0.5, min_players=2, max_player_height_ratio=0.2)

    # Frame height is 100.
    # Player 1 height: 30 (ratio = 0.3 > 0.2) -> Close up!
    # Player 2 height: 10 (ratio = 0.1)
    dummy_det = DummyDetector(
        boxes=[[10, 10, 20, 40], [30, 30, 40, 40]],
        classes=[0, 0]
    )

    phase = detector.evaluate(green_frame(), dummy_det)
    assert not phase.in_play
    assert phase.metrics["reason"] == "close_up"
    assert phase.metrics["max_player_height_ratio"] == 0.3


def test_in_play_returns_reusable_prediction():
    detector = PlayPhaseDetector(min_green_ratio=0.5, min_players=2, max_player_height_ratio=0.2)

    # Both players have height ratio 0.1 <= 0.2
    dummy_det = DummyDetector(
        boxes=[[10, 10, 20, 20], [30, 30, 40, 40]],
        classes=[0, 0]
    )

    phase = detector.evaluate(green_frame(), dummy_det)
    assert phase.in_play
    assert phase.metrics["reason"] == "in_play"
    assert phase.metrics["num_players"] == 2
    # La prédiction doit remonter pour que le pipeline ne redétecte pas
    assert dummy_det.calls == 1
    assert phase.prediction is not None
    assert len(phase.prediction.boxes) == 2


def test_non_person_classes_ignored():
    detector = PlayPhaseDetector(min_green_ratio=0.5, min_players=2, max_player_height_ratio=0.5)

    # Un seul person, le reste est du bruit COCO (sports ball, tv)
    dummy_det = DummyDetector(
        boxes=[[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]],
        classes=[0, 32, 62]
    )

    phase = detector.evaluate(green_frame(), dummy_det)
    assert not phase.in_play
    assert phase.metrics["num_players"] == 1
