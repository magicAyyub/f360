from .base import Detector, Prediction

__all__ = ['Detector', 'Prediction', 'YoloDetector', 'RTDetrDetector']


def __getattr__(name: str):
    if name == 'YoloDetector':
        from .yolo_detector import YoloDetector

        return YoloDetector
    if name == 'RTDetrDetector':
        from .rtdetr_detector import RTDetrDetector

        return RTDetrDetector
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return __all__
