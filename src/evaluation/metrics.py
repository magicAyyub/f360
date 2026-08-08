from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import motmetrics as mm
import numpy as np
from trackeval.metrics import HOTA

from .mot import COLUMNS, frame_range, group_by_frame


@dataclass
class TrackingMetrics:
    """Tracking scores for one sequence.

    HOTA/DetA/AssA are averaged over TrackEval's localisation thresholds, so they
    do not depend on iou_threshold; the CLEAR family and IDF1 do.
    """

    hota: float
    deta: float
    assa: float
    mota: float
    motp: float
    idf1: float
    precision: float
    recall: float
    id_switches: int
    false_positives: int
    misses: int
    num_gt_boxes: int
    num_pred_boxes: int
    num_frames: int

    def as_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def iou_matrix(gt_tlwh: np.ndarray, pred_tlwh: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of [left, top, width, height] boxes."""
    if len(gt_tlwh) == 0 or len(pred_tlwh) == 0:
        return np.zeros((len(gt_tlwh), len(pred_tlwh)))

    gt = np.asarray(gt_tlwh, dtype=float)
    pred = np.asarray(pred_tlwh, dtype=float)

    gt_x2, gt_y2 = gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]
    pred_x2, pred_y2 = pred[:, 0] + pred[:, 2], pred[:, 1] + pred[:, 3]

    left = np.maximum(gt[:, None, 0], pred[None, :, 0])
    top = np.maximum(gt[:, None, 1], pred[None, :, 1])
    right = np.minimum(gt_x2[:, None], pred_x2[None, :])
    bottom = np.minimum(gt_y2[:, None], pred_y2[None, :])

    overlap = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    union = (gt[:, 2] * gt[:, 3])[:, None] + (pred[:, 2] * pred[:, 3])[None, :] - overlap

    return np.where(union > 0, overlap / np.where(union > 0, union, 1), 0.0)


def _dense_ids(rows_by_frame: Dict[int, np.ndarray], frames: np.ndarray) -> Tuple[List[np.ndarray], int]:
    """Remap track ids to contiguous 0-based indices for HOTA.

    TrackEval indexes its accumulators by id value, so raw ids allocate a matrix
    sized by the largest id rather than by the number of distinct tracks: a
    ByteTrack id of 5000 costs 200 MB against 13 KB here. Unused rows also divide
    zero by zero, which floods the run with warnings. Scores are unaffected.
    """
    all_ids = np.concatenate([rows_by_frame[f][:, 1] for f in frames]) if len(frames) else np.zeros(0)
    lookup = {raw: index for index, raw in enumerate(np.unique(all_ids))}

    per_frame = [
        np.array([lookup[raw] for raw in rows_by_frame[f][:, 1]], dtype=int)
        for f in frames
    ]
    return per_frame, len(lookup)


CLEAR_METRICS = ['mota', 'motp', 'idf1', 'precision', 'recall',
                 'num_switches', 'num_false_positives', 'num_misses']


def _prepare(gt: np.ndarray, pred: np.ndarray, iou_threshold: float):
    """Build the per-sequence intermediates both metric families consume."""
    gt = np.asarray(gt, dtype=float).reshape(-1, len(COLUMNS))
    pred = np.asarray(pred, dtype=float).reshape(-1, len(COLUMNS))

    frames = frame_range(gt, pred)
    gt_by_frame = group_by_frame(gt, frames)
    pred_by_frame = group_by_frame(pred, frames)

    similarities = [
        iou_matrix(gt_by_frame[f][:, 2:6], pred_by_frame[f][:, 2:6])
        for f in frames
    ]

    accumulator = mm.MOTAccumulator(auto_id=False)
    for index, frame in enumerate(frames):
        # motmetrics raisonne en distance: 1 - IoU, et écarte tout ce qui dépasse le seuil
        distances = 1.0 - similarities[index]
        distances[similarities[index] < iou_threshold] = np.nan

        accumulator.update(
            gt_by_frame[frame][:, 1].astype(int).tolist(),
            pred_by_frame[frame][:, 1].astype(int).tolist(),
            distances,
            frameid=int(frame),
        )

    gt_ids, num_gt_ids = _dense_ids(gt_by_frame, frames)
    pred_ids, num_pred_ids = _dense_ids(pred_by_frame, frames)

    hota_raw = HOTA().eval_sequence({
        'num_timesteps': len(frames),
        'num_gt_ids': num_gt_ids,
        'num_tracker_ids': num_pred_ids,
        'num_gt_dets': int(sum(len(ids) for ids in gt_ids)),
        'num_tracker_dets': int(sum(len(ids) for ids in pred_ids)),
        'gt_ids': gt_ids,
        'tracker_ids': pred_ids,
        'similarity_scores': similarities,
    })

    counts = {
        'num_gt_boxes': len(gt),
        'num_pred_boxes': len(pred),
        'num_frames': len(frames),
    }
    return accumulator, hota_raw, counts


def _summarise(clear_row, hota_raw: Dict[str, np.ndarray], counts: Dict[str, int]) -> TrackingMetrics:
    motp = clear_row['motp']
    return TrackingMetrics(
        # HOTA sort un tableau par seuil de localisation, le scalaire publié est la moyenne
        hota=float(np.mean(hota_raw['HOTA'])),
        deta=float(np.mean(hota_raw['DetA'])),
        assa=float(np.mean(hota_raw['AssA'])),
        mota=float(clear_row['mota']),
        # motp sort en distance moyenne, on le rend en IoU pour rester lisible
        motp=float(1.0 - motp) if np.isfinite(motp) else 0.0,
        idf1=float(clear_row['idf1']),
        precision=float(clear_row['precision']),
        recall=float(clear_row['recall']),
        id_switches=int(clear_row['num_switches']),
        false_positives=int(clear_row['num_false_positives']),
        misses=int(clear_row['num_misses']),
        **counts,
    )


def evaluate(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5) -> TrackingMetrics:
    """Score predicted tracks against ground truth, both in MOT row format."""
    accumulator, hota_raw, counts = _prepare(gt, pred, iou_threshold)

    summary = mm.metrics.create().compute(accumulator, metrics=CLEAR_METRICS).iloc[0]
    return _summarise(summary, hota_raw, counts)


def evaluate_many(
    sequences: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    iou_threshold: float = 0.5,
) -> Tuple[Dict[str, TrackingMetrics], TrackingMetrics]:
    """Score several sequences and combine them the way the benchmarks do.

    The combined score is not the mean of the per-sequence scores: TrackEval sums
    the underlying detection counts and weights association by true positives, so
    long sequences carry more weight. Averaging would give a different number that
    matches no published table.
    """
    if not sequences:
        raise ValueError("no sequences to evaluate")

    names = list(sequences)
    prepared = {name: _prepare(*sequences[name], iou_threshold) for name in names}

    summaries = mm.metrics.create().compute_many(
        [prepared[name][0] for name in names],
        names=names,
        metrics=CLEAR_METRICS,
        generate_overall=True,
    )

    per_sequence = {
        name: _summarise(summaries.loc[name], prepared[name][1], prepared[name][2])
        for name in names
    }

    combined_hota = HOTA().combine_sequences({name: prepared[name][1] for name in names})
    total_counts = {
        key: sum(prepared[name][2][key] for name in names)
        for key in ('num_gt_boxes', 'num_pred_boxes', 'num_frames')
    }
    overall = _summarise(summaries.loc['OVERALL'], combined_hota, total_counts)

    return per_sequence, overall
