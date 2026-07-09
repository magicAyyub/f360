# Football Analytics Project Test Guide

This guide provides instructions to run automated tests, execute different pipeline tracking configurations, and run performance benchmarks.

---

## 1. Running Automated Unit Tests

Automated tests are built with `pytest` and mock the model weights (YOLO, ByteTrack, SAM2) to execute instantly and reliably.

Run the test suite using `uv`:
```bash
uv run pytest
```

### Test Coverage
* **Play Detection Heuristics:** [test_play_detector.py](tests/test_play_detector.py) (verifies green ratio, player counts, and player height checks).
* **ByteTrack Tracker:** [test_bytetrack_tracker.py](tests/test_bytetrack_tracker.py) (verifies track updates, object formatting, and class logic).
* **SAM2 Segmenter:** [test_sam_segmenter.py](tests/test_sam_segmenter.py) (verifies target mask creation, dimensions, and index mapping).

---

## 2. Pipeline Execution Commands

The processing pipeline is executed using the `run-match` package entry point.

### A. Default Pipeline (Fast Track)
Processes the match clip using **YOLOv5** and **ByteTrack** for tracking without segmentation masks. This is the recommended mode for standard analytics.
```bash
uv run run-match --start-time 315 --end-time 325 --output-video outputs/fast_run.mp4
```

### B. SAM2 Player Segmentation Pipeline (Visual Enhancement)
Processes the match clip using **YOLOv5**, **ByteTrack**, and **SAM2** to generate transparent, color-coded segmentation masks for every player.
```bash
uv run run-match --start-time 315 --end-time 325 --tracker-type bytetrack_sam --output-video outputs/sam_run.mp4
```

### C. Play Phase Filter
Only processes and outputs frames that are classified as "PLAYING" (bypassing close-ups, replays, and low-green shots to speed up execution).
```bash
uv run run-match --start-time 315 --end-time 325 --only-in-play --output-video outputs/only_play.mp4
```

---

## 3. Profiling Performance and Component Latency

To profile memory usage and identify speed bottlenecks across frames, execute the benchmarking script:
```bash
uv run python src/analytics/benchmark.py
```

This script will run on your active compute device (CPU or MPS on macOS) and print:
* Average throughput (FPS) and latency (ms) per frame.
* Memory usage (RSS RAM).
* Latency breakdown for video decoding, play phase heuristics, YOLO, ByteTrack, and SAM2.
