from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import cv2
import numpy as np

from src.detection.base import Detector, Prediction


@dataclass
class PlayPhase:
    """Verdict for a single frame, carrying the detections used to reach it.

    prediction is None when the green-ratio test short-circuited before running
    the detector; callers should reuse it rather than detecting a second time.
    """

    in_play: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    prediction: Optional[Prediction] = None


class PlayPhaseDetector:
    """Heuristic engine to detect if a football match video frame is in the playing phase."""

    PERSON_CLASS = 0

    def __init__(
        self,
        min_green_ratio: float = 0.50,
        min_players: int = 5,
        max_player_height_ratio: float = 0.25,
    ) -> None:
        self.min_green_ratio = min_green_ratio
        self.min_players = min_players
        self.max_player_height_ratio = max_player_height_ratio

    def compute_green_ratio(self, frame: np.ndarray) -> float:
        """Calculate the ratio of green pixels in the frame (pitch detection)."""
        # Redimensionnement pour accélérer le traitement
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (320, int(320 * h / w)))

        # Conversion en HSV pour le seuillage de couleur
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

        # Plage de couleur pour la pelouse verte (Hue, Saturation, Value)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = float(np.sum(mask > 0) / mask.size)
        return green_ratio

    def evaluate(self, frame: np.ndarray, detector: Optional[Detector] = None) -> PlayPhase:
        """Determine whether the frame shows active wide-angle play."""
        # Étape 1 : vérification rapide du ratio de vert, avant tout appel au détecteur
        green_ratio = self.compute_green_ratio(frame)
        if green_ratio < self.min_green_ratio:
            return PlayPhase(False, {"green_ratio": green_ratio, "reason": "low_green_ratio"})

        # Si aucun détecteur n'est fourni, on se fie uniquement au ratio de vert
        if detector is None:
            return PlayPhase(True, {
                "green_ratio": green_ratio,
                "reason": "high_green_ratio_no_detector",
            })

        # Étape 2 : détection des joueurs (classe person = 0 dans COCO)
        prediction = detector.predict(frame)
        person_boxes = prediction.boxes[prediction.classes == self.PERSON_CLASS]
        num_players = len(person_boxes)

        if num_players < self.min_players:
            return PlayPhase(False, {
                "green_ratio": green_ratio,
                "num_players": num_players,
                "reason": "insufficient_players",
            }, prediction)

        # Étape 3 : vérification de la taille des boîtes (filtre les gros plans)
        frame_height = frame.shape[0]
        box_heights = (person_boxes[:, 3] - person_boxes[:, 1]) / frame_height
        max_height = float(np.max(box_heights))

        if max_height > self.max_player_height_ratio:
            return PlayPhase(False, {
                "green_ratio": green_ratio,
                "num_players": num_players,
                "max_player_height_ratio": max_height,
                "reason": "close_up",
            }, prediction)

        return PlayPhase(True, {
            "green_ratio": green_ratio,
            "num_players": num_players,
            "max_player_height_ratio": max_height,
            "reason": "in_play",
        }, prediction)
