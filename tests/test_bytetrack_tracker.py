import sys
import types

import numpy as np


def _make_prediction(boxes, scores, classes):
    from src.detection.base import Prediction

    return Prediction(boxes=np.asarray(boxes), scores=np.asarray(scores), classes=np.asarray(classes))


class DummyDetections:
    def __init__(self, xyxy, confidence, class_id, tracker_id):
        self.xyxy = np.asarray(xyxy)
        self.confidence = np.asarray(confidence)
        self.class_id = np.asarray(class_id)
        self.tracker_id = np.asarray(tracker_id)


class DummyDetectionsClass:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id


def test_supervision_byte_track_numpy_output(monkeypatch):
    class DummyTracker:
        def __init__(self, *a, **k):
            pass

        def update_with_detections(self, d):
            return DummyDetections(
                xyxy=[[10, 20, 30, 40], [50, 60, 70, 80]],
                confidence=[0.9, 0.8],
                class_id=[0, 0],
                tracker_id=[1, 2]
            )

    dummy_mod = types.SimpleNamespace(ByteTrack=DummyTracker, Detections=DummyDetectionsClass)
    monkeypatch.setitem(sys.modules, 'supervision', dummy_mod)

    from importlib import reload
    import src.tracking.bytetrack_tracker as bytetrack_tracker_mod
    reload(bytetrack_tracker_mod)
    from src.tracking.bytetrack_tracker import ByteTrackTracker

    pred = _make_prediction([[10, 20, 30, 40], [50, 60, 70, 80]], [0.9, 0.8], [0, 0])
    tracker = ByteTrackTracker()
    tracks = tracker.update(pred)

    assert len(tracks) == 2
    assert tracks[0].track_id == 1
    assert np.allclose(tracks[0].box, [10, 20, 30, 40])
    assert abs(tracks[0].score - 0.9) < 1e-6


def test_supervision_byte_track_iterable_output(monkeypatch):
    class DummyTracker:
        def __init__(self, *a, **k):
            pass

        def update_with_detections(self, d):
            return DummyDetections(
                xyxy=[[5, 6, 15, 16]],
                confidence=[0.75],
                class_id=[1],
                tracker_id=[3]
            )

    dummy_mod = types.SimpleNamespace(ByteTrack=DummyTracker, Detections=DummyDetectionsClass)
    monkeypatch.setitem(sys.modules, 'supervision', dummy_mod)

    from importlib import reload
    import src.tracking.bytetrack_tracker as bytetrack_tracker_mod
    reload(bytetrack_tracker_mod)
    from src.tracking.bytetrack_tracker import ByteTrackTracker

    pred = _make_prediction([[5, 6, 15, 16]], [0.75], [1])
    tracker = ByteTrackTracker()
    tracks = tracker.update(pred)

    assert len(tracks) == 1
    assert tracks[0].track_id == 3
    assert np.allclose(tracks[0].box, [5, 6, 15, 16])
    assert abs(tracks[0].score - 0.75) < 1e-6



def test_no_supervision_installed(monkeypatch):
    # Ensure supervision is not available
    monkeypatch.setitem(sys.modules, 'supervision', None)

    from importlib import reload
    from src.tracking import bytetrack_tracker
    from pytest import raises

    with raises(ImportError):
        reload(bytetrack_tracker)

