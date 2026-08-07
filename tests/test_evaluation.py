import numpy as np
import pytest

from src.evaluation import evaluate, iou_matrix, load_mot, save_mot, tracks_to_mot, xyxy_to_tlwh
from src.evaluation.metrics import _dense_ids
from src.tracking.base import Track


def rows(*entries):
    """Build MOT rows from (frame, track_id, x, y) tuples with a fixed 10x20 box."""
    return np.array([[f, tid, x, y, 10.0, 20.0, 1.0] for f, tid, x, y in entries], dtype=float)


def test_xyxy_to_tlwh():
    converted = xyxy_to_tlwh(np.array([[10, 20, 40, 80]]))
    assert np.allclose(converted, [[10, 20, 30, 60]])


def test_tracks_to_mot_uses_sequence_frame_number():
    tracks = [
        Track(track_id=7, box=np.array([10, 20, 40, 80]), score=0.9, class_id=0),
        Track(track_id=9, box=np.array([0, 0, 10, 10]), score=0.4, class_id=0),
    ]
    converted = tracks_to_mot(frame=1, tracks=tracks)

    assert converted.shape == (2, 7)
    assert np.allclose(converted[:, 0], [1, 1])
    assert np.allclose(converted[:, 1], [7, 9])
    assert np.allclose(converted[0, 2:6], [10, 20, 30, 60])
    assert np.allclose(converted[:, 6], [0.9, 0.4])


def test_tracks_to_mot_empty():
    assert tracks_to_mot(frame=3, tracks=[]).shape == (0, 7)


def test_mot_round_trip(tmp_path):
    original = rows((1, 1, 0, 0), (1, 2, 50, 50), (2, 1, 5, 0))
    path = tmp_path / 'gt.txt'
    save_mot(original, path)

    assert np.allclose(load_mot(path), original)


def test_load_mot_drops_ignored_rows(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text("1,1,0,0,10,20,1\n1,2,50,50,10,20,0\n")

    loaded = load_mot(path)
    assert len(loaded) == 1
    assert loaded[0, 1] == 1


def test_iou_matrix():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(box, np.array([[100.0, 100.0, 10.0, 10.0]]))[0, 0] == pytest.approx(0.0)

    # Recouvrement de moitié: intersection 50, union 150
    half = iou_matrix(box, np.array([[5.0, 0.0, 10.0, 10.0]]))[0, 0]
    assert half == pytest.approx(50 / 150)


def test_iou_matrix_handles_empty():
    assert iou_matrix(np.zeros((0, 4)), np.array([[0.0, 0.0, 1.0, 1.0]])).shape == (0, 1)


def test_perfect_prediction():
    gt = rows((1, 1, 0, 0), (1, 2, 50, 50), (2, 1, 5, 0), (2, 2, 55, 50))
    result = evaluate(gt, gt.copy())

    assert result.hota == pytest.approx(1.0)
    assert result.deta == pytest.approx(1.0)
    assert result.assa == pytest.approx(1.0)
    assert result.mota == pytest.approx(1.0)
    assert result.idf1 == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.id_switches == 0
    assert result.false_positives == 0
    assert result.misses == 0
    assert result.num_frames == 2


def test_empty_prediction():
    gt = rows((1, 1, 0, 0), (2, 1, 5, 0))
    result = evaluate(gt, np.zeros((0, 7)))

    assert result.recall == pytest.approx(0.0)
    assert result.misses == 2
    assert result.hota == pytest.approx(0.0)
    assert result.mota == pytest.approx(0.0)


def test_half_the_detections_missed():
    gt = rows((1, 1, 0, 0), (1, 2, 50, 50), (2, 1, 5, 0), (2, 2, 55, 50))
    pred = rows((1, 1, 0, 0), (2, 1, 5, 0))

    result = evaluate(gt, pred)
    assert result.recall == pytest.approx(0.5)
    assert result.precision == pytest.approx(1.0)
    assert result.misses == 2
    assert result.false_positives == 0
    # MOTA = 1 - (FN + FP + IDSW) / num_gt = 1 - 2/4
    assert result.mota == pytest.approx(0.5)


def test_single_id_switch():
    # Un seul objet sur 4 frames, le tracker change d'id à mi-parcours
    gt = rows((1, 1, 0, 0), (2, 1, 0, 0), (3, 1, 0, 0), (4, 1, 0, 0))
    pred = rows((1, 1, 0, 0), (2, 1, 0, 0), (3, 2, 0, 0), (4, 2, 0, 0))

    result = evaluate(gt, pred)

    # La détection est parfaite, seule l'association casse
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.deta == pytest.approx(1.0)
    assert result.id_switches == 1

    # MOTA = 1 - 1/4, et chaque id ne couvre que la moitié de la trajectoire
    assert result.mota == pytest.approx(0.75)
    assert result.idf1 == pytest.approx(0.5)
    assert result.assa == pytest.approx(0.5)
    assert result.hota == pytest.approx(np.sqrt(0.5))


def test_dense_ids_are_contiguous_and_zero_based():
    """HOTA allocates by largest id value, so ids are compacted before it runs."""
    frames = np.array([1, 2])
    rows_by_frame = {
        1: rows((1, 4207, 0, 0), (1, 991, 50, 50)),
        2: rows((2, 991, 5, 0)),
    }

    per_frame, count = _dense_ids(rows_by_frame, frames)

    assert count == 2
    assert sorted(np.concatenate(per_frame).tolist()) == [0, 0, 1]
    # 991 < 4207, donc l'ordre trié fixe le mapping
    assert per_frame[0].tolist() == [1, 0]
    assert per_frame[1].tolist() == [0]


def test_scores_are_independent_of_id_values():
    dense_gt = rows((1, 0, 0, 0), (1, 1, 50, 50), (2, 0, 5, 0), (2, 1, 55, 50))
    dense_pred = rows((1, 0, 0, 0), (1, 1, 50, 50), (2, 1, 5, 0), (2, 0, 55, 50))

    sparse_gt = dense_gt.copy()
    sparse_pred = dense_pred.copy()
    for original, replacement in ((0, 4207), (1, 991)):
        sparse_gt[dense_gt[:, 1] == original, 1] = replacement
        sparse_pred[dense_pred[:, 1] == original, 1] = replacement

    assert evaluate(dense_gt, dense_pred).as_dict() == evaluate(sparse_gt, sparse_pred).as_dict()


def test_no_nan_in_reported_scores():
    gt = rows((1, 3, 0, 0), (1, 9, 50, 50), (2, 3, 5, 0), (2, 9, 55, 50))
    pred = rows((1, 12, 0, 0), (1, 40, 50, 50), (2, 40, 5, 0), (2, 12, 55, 50))

    for name, value in evaluate(gt, pred).as_dict().items():
        assert np.isfinite(value), f"{name} is not finite"


def test_frames_without_ground_truth_count_as_false_positives():
    gt = rows((1, 1, 0, 0))
    pred = rows((1, 1, 0, 0), (2, 1, 0, 0))

    result = evaluate(gt, pred)
    assert result.false_positives == 1
    assert result.num_frames == 2
