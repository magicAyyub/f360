import sys
import types

import numpy as np


def _make_prediction(boxes, scores, classes):
    from src.detection.base import Prediction

    return Prediction(boxes=np.asarray(boxes), scores=np.asarray(scores), classes=np.asarray(classes))


def test_supervision_byte_track_numpy_output(monkeypatch):
    class DummyTracker:
        def __init__(self, *a, **k):
            pass

        def update(self, d):
            # return array with columns: id,x1,y1,x2,y2,score
            return np.array([[1, 10, 20, 30, 40, 0.9], [2, 50, 60, 70, 80, 0.8]])

    dummy_mod = types.SimpleNamespace(ByteTrack=DummyTracker)
    monkeypatch.setitem(sys.modules, 'supervision', dummy_mod)

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

        def update(self, d):
            # return a list of tuples (id,x1,y1,x2,y2,score)
            return [(3, 5, 6, 15, 16, 0.75)]

    dummy_mod = types.SimpleNamespace(ByteTrack=DummyTracker)
    monkeypatch.setitem(sys.modules, 'supervision', dummy_mod)

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
    try:
        del sys.modules['supervision']
    except KeyError:
        pass

    from importlib import reload
    from src.tracking import bytetrack_tracker

    reload(bytetrack_tracker)

    from src.detection.base import Prediction

    pred = Prediction(boxes=np.zeros((0, 4)), scores=np.zeros((0,)), classes=np.zeros((0,), dtype=int))

    from pytest import raises

    with raises(ImportError):
        bytetrack_tracker.ByteTrackTracker()
