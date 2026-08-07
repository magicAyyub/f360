from .base import Detector, Prediction

__all__ = ['Detector', 'Prediction', 'YoloDetector']


def __getattr__(name: str):
    if name == 'YoloDetector':
        from .yolo_detector import YoloDetector

        return YoloDetector
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return __all__
