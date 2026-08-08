import sys
import types
from importlib import import_module, reload

import numpy as np
import pytest

import src.tracking
import src.tracking.bytetrack_tracker as bytetrack_tracker
from src.detection.base import Prediction


@pytest.fixture(autouse=True)
def restore_real_supervision():
    """Put the module back after tests that swap out supervision.

    Reloading bytetrack_tracker against a fake supervision leaves the reloaded
    module holding that fake, which monkeypatch cannot undo. Without this, every
    later test in the session builds a tracker that returns canned boxes.
    """
    real = import_module('supervision')
    yield
    sys.modules['supervision'] = real
    reload(bytetrack_tracker)
    reload(src.tracking)
    for name in ('src.pipeline.run_sequence', 'src.pipeline.run_match'):
        if name in sys.modules:
            reload(sys.modules[name])


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


def fake_supervision(tracked: DummyDetections) -> types.SimpleNamespace:
    class DummyTracker:
        def __init__(self, *args, **kwargs):
            pass

        def update_with_detections(self, detections):
            return tracked

    return types.SimpleNamespace(ByteTrack=DummyTracker, Detections=DummyDetectionsClass)


def make_prediction(boxes, scores, classes) -> Prediction:
    return Prediction(boxes=np.asarray(boxes), scores=np.asarray(scores), classes=np.asarray(classes))


def test_converts_tracked_detections_to_tracks(monkeypatch):
    tracked = DummyDetections(
        xyxy=[[10, 20, 30, 40], [50, 60, 70, 80]],
        confidence=[0.9, 0.8],
        class_id=[0, 0],
        tracker_id=[1, 2],
    )
    # setattr plutôt qu'un reload: pytest le défait tout seul
    monkeypatch.setattr(bytetrack_tracker, 'sv', fake_supervision(tracked))

    tracks = bytetrack_tracker.ByteTrackTracker().update(
        make_prediction([[10, 20, 30, 40], [50, 60, 70, 80]], [0.9, 0.8], [0, 0])
    )

    assert len(tracks) == 2
    assert tracks[0].track_id == 1
    assert np.allclose(tracks[0].box, [10, 20, 30, 40])
    assert tracks[0].score == pytest.approx(0.9)


def test_keeps_class_ids(monkeypatch):
    tracked = DummyDetections(
        xyxy=[[5, 6, 15, 16]],
        confidence=[0.75],
        class_id=[1],
        tracker_id=[3],
    )
    monkeypatch.setattr(bytetrack_tracker, 'sv', fake_supervision(tracked))

    tracks = bytetrack_tracker.ByteTrackTracker().update(make_prediction([[5, 6, 15, 16]], [0.75], [1]))

    assert len(tracks) == 1
    assert tracks[0].track_id == 3
    assert tracks[0].class_id == 1
    assert np.allclose(tracks[0].box, [5, 6, 15, 16])


def test_import_fails_without_supervision(monkeypatch):
    monkeypatch.setitem(sys.modules, 'supervision', None)

    with pytest.raises(ImportError):
        reload(bytetrack_tracker)


def test_real_tracker_is_restored_between_tests():
    """Guards the fixture above: a leaked fake would surface right here."""
    import supervision as sv

    assert bytetrack_tracker.sv is sv
    assert src.tracking.ByteTrackTracker is bytetrack_tracker.ByteTrackTracker
