# Quick Test Guide

## Unit tests

Tests mock the model weights so they run instantly and without downloads.

```bash
uv run pytest
```

Coverage:

* Video reader: [test_video_reader.py](tests/test_video_reader.py) verifies frame rate propagation through the sampler chain, and that seeking agrees with full decoding on frame ids and timestamps.
* Play detection heuristics: [test_play_detector.py](tests/test_play_detector.py) verifies the green ratio, player counts, close-up rejection, and that the green short-circuit avoids running the detector.
* ByteTrack tracker: [test_bytetrack_tracker.py](tests/test_bytetrack_tracker.py) verifies track updates and prediction conversion.

## Pipeline

The pipeline runs through the `run-match` entry point. Detection is restricted to the COCO person class, and the output video is written at the source frame rate divided by the stride.

Standard run over a ten second window:

```bash
uv run run-match --start-time 315 --end-time 325 --output-video outputs/run.mp4
```

Half the frames, same wall-clock duration in the output:

```bash
uv run run-match --start-time 315 --end-time 325 --stride 2 --output-video outputs/run_stride2.mp4
```

Keep only frames classified as in play, skipping replays and close-ups. The tracker resets at each play-phase boundary, so track ids are not carried across a cut:

```bash
uv run run-match --start-time 315 --end-time 325 --only-in-play --output-video outputs/in_play.mp4
```

## Evaluation

The pipeline can dump its tracks in MOTChallenge format, which is what the metrics read and what SoccerNet ground truth already ships as:

```bash
uv run run-match --start-time 315 --end-time 325 --output-tracks outputs/pred.txt
```

Score a prediction file against ground truth:

```bash
uv run eval-tracks path/to/gt.txt outputs/pred.txt
```

HOTA, DetA and AssA come from TrackEval and average over its localisation thresholds, so they ignore `--iou-threshold`. MOTA, IDF1, precision and recall come from motmetrics and depend on it. Frame numbers are 1-based and local to the evaluated window, not source video frame ids, so ground truth must be labelled over the same window the pipeline ran on.

Scoring a prediction file against itself returns 1.0 everywhere, which is the quickest check that a new ground truth file parses the way you expect.

## Performance profiling

```bash
uv run python src/analytics/benchmark.py
```

Reports throughput, per-frame latency, RSS memory, and a latency breakdown across decode, green-ratio heuristic, YOLO and ByteTrack. It also reports detector calls per frame, which must stay at 1.00: anything higher means the play detector and the pipeline are each running inference instead of sharing one prediction.
