from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def evaluate(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5) -> TrackingMetrics:
    """Score predicted tracks against ground truth, both in MOT row format."""
    gt = np.asarray(gt, dtype=float).reshape(-1, len(COLUMNS))
    pred = np.asarray(pred, dtype=float).reshape(-1, len(COLUMNS))

    frames = frame_range(gt, pred)
    gt_by_frame = group_by_frame(gt, frames)
    pred_by_frame = group_by_frame(pred, frames)

    similarities = [
        iou_matrix(gt_by_frame[f][:, 2:6], pred_by_frame[f][:, 2:6])
        for f in frames
    ]

    clear = _clear_metrics(gt_by_frame, pred_by_frame, frames, similarities, iou_threshold)
    hota = _hota_metrics(gt_by_frame, pred_by_frame, frames, similarities)

    return TrackingMetrics(
        hota=hota['HOTA'],
        deta=hota['DetA'],
        assa=hota['AssA'],
        mota=clear['mota'],
        motp=clear['motp'],
        idf1=clear['idf1'],
        precision=clear['precision'],
        recall=clear['recall'],
        id_switches=clear['id_switches'],
        false_positives=clear['false_positives'],
        misses=clear['misses'],
        num_gt_boxes=len(gt),
        num_pred_boxes=len(pred),
        num_frames=len(frames),
    )


def _clear_metrics(gt_by_frame, pred_by_frame, frames, similarities, iou_threshold) -> Dict[str, float]:
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

    summary = mm.metrics.create().compute(
        accumulator,
        metrics=['mota', 'motp', 'idf1', 'precision', 'recall',
                 'num_switches', 'num_false_positives', 'num_misses'],
    ).iloc[0]

    motp = summary['motp']
    return {
        'mota': float(summary['mota']),
        # motp sort en distance moyenne, on le rend en IoU pour rester lisible
        'motp': float(1.0 - motp) if np.isfinite(motp) else 0.0,
        'idf1': float(summary['idf1']),
        'precision': float(summary['precision']),
        'recall': float(summary['recall']),
        'id_switches': int(summary['num_switches']),
        'false_positives': int(summary['num_false_positives']),
        'misses': int(summary['num_misses']),
    }


def _hota_metrics(gt_by_frame, pred_by_frame, frames, similarities) -> Dict[str, float]:
    gt_ids, num_gt_ids = _dense_ids(gt_by_frame, frames)
    pred_ids, num_pred_ids = _dense_ids(pred_by_frame, frames)

    scores = HOTA().eval_sequence({
        'num_timesteps': len(frames),
        'num_gt_ids': num_gt_ids,
        'num_tracker_ids': num_pred_ids,
        'num_gt_dets': int(sum(len(ids) for ids in gt_ids)),
        'num_tracker_dets': int(sum(len(ids) for ids in pred_ids)),
        'gt_ids': gt_ids,
        'tracker_ids': pred_ids,
        'similarity_scores': similarities,
    })

    # HOTA sort un tableau par seuil de localisation, le scalaire publié est la moyenne
    return {key: float(np.mean(scores[key])) for key in ('HOTA', 'DetA', 'AssA')}
