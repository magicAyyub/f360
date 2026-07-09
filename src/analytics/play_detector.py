from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from src.detection.base import Detector


class PlayPhaseDetector:
    """Heuristic engine to detect if a football match video frame is in the playing phase."""

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

    def is_in_play(
        self,
        frame: np.ndarray,
        detector: Optional[Detector] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if the frame is active wide-angle play.
        
        Returns:
            A tuple of (is_playing_bool, debug_metrics_dict).
        """
        # Étape 1 : Vérification rapide du ratio de vert
        green_ratio = self.compute_green_ratio(frame)
        if green_ratio < self.min_green_ratio:
            return False, {
                "green_ratio": green_ratio,
                "reason": "low_green_ratio",
            }

        # Si aucun détecteur n'est fourni, on se fie uniquement au ratio de vert
        if detector is None:
            return True, {
                "green_ratio": green_ratio,
                "reason": "high_green_ratio_no_detector",
            }

        # Étape 2 : Détection des joueurs (classe person = 0 dans COCO/YOLO)
        pred = detector.predict(frame)
        person_idx = (pred.classes == 0)
        person_boxes = pred.boxes[person_idx]
        num_players = len(person_boxes)

        if num_players < self.min_players:
            return False, {
                "green_ratio": green_ratio,
                "num_players": num_players,
                "reason": "insufficient_players",
            }

        # Étape 3 : Vérification de la taille maximale des boîtes (filtre les gros plans)
        frame_height = frame.shape[0]
        box_heights = (person_boxes[:, 3] - person_boxes[:, 1]) / frame_height if num_players > 0 else np.array([])
        max_height = float(np.max(box_heights)) if len(box_heights) > 0 else 0.0

        if max_height > self.max_player_height_ratio:
            return False, {
                "green_ratio": green_ratio,
                "num_players": num_players,
                "max_player_height_ratio": max_height,
                "reason": "close_up",
            }

        return True, {
            "green_ratio": green_ratio,
            "num_players": num_players,
            "max_player_height_ratio": max_height,
            "reason": "in_play",
        }
