# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding standards

Follow `.agents/skills/developer-standards/SKILL.md`. It is the user's own file and takes precedence over defaults. The points that bite most often: inline comments may be French while code, names and docstrings stay English; no emoji, no decorative separators, no `1.` `2.` enumerations; be critical rather than agreeable when a simpler or more standard option exists.

## Commands

```bash
uv run pytest                              # full suite
uv run pytest tests/test_evaluation.py -q  # one file
uv run pytest -k oracle                    # one test by name
uv run pytest -W error::RuntimeWarning     # escalate nan divisions to failures
```

`pytest` lives in the `dev` dependency group, not `[project.optional-dependencies]`. If it is ever moved back, `uv run pytest` silently falls through to whatever `pytest` is on PATH and the suite runs against a different interpreter and package set while still passing.

```bash
uv run run-match --start-time 315 --end-time 325 --output-video outputs/run.mp4
uv run run-match --start-time 315 --end-time 325 --output-tracks outputs/pred.txt
uv run python src/analytics/benchmark.py   # latency breakdown, not accuracy
```

```bash
uv run run-sequence data/SoccerNet/test --output-dir outputs/soccernet
uv run run-sequence data/SoccerNet/test --output-dir outputs/soccernet-oracle --oracle
uv run eval-dataset data/SoccerNet/test outputs/soccernet --show-sequences
uv run eval-tracks data/SoccerNet/test/SNMOT-116 outputs/soccernet/SNMOT-116.txt
```

`quick_test_guide.md` covers the SoccerNet download, which needs resume and a stall timeout because the server drops connections mid-transfer.

## Architecture

Two pipelines share the same components. `src/pipeline/run_match.py` runs over a broadcast video file, `src/pipeline/run_sequence.py` over MOT-format image sequences for evaluation. Anything added to one usually belongs in both.

**Video layer** (`src/video`) is a chain of composable iterables yielding `Frame` objects: `VideoReader` or `ImageSequenceReader`, then optionally `TimeWindowFilter` and `FrameSampler`. Every stage exposes an `fps` property, and `FrameSampler` divides it by its stride. The pipeline reads `reader.fps` at the end of the chain and feeds it to both `ByteTrackTracker(frame_rate=...)` and the video writer. Never hardcode a frame rate: the source runs at 58.8 fps, a hardcoded 30 made `lost_track_buffer` mean half what it claimed and wrote every output video at half speed.

**Detection and tracking** (`src/detection`, `src/tracking`) sit behind `Detector` and `Tracker` ABCs exchanging `Prediction` and `Track`. Detection is restricted to the COCO person class at the call site; without that filter YOLO feeds sports ball, frisbee and tv boxes into ByteTrack as player tracks.

**Play phase** (`src/analytics/play_detector.py`) returns a `PlayPhase` carrying the `Prediction` it used. The pipeline reuses it rather than detecting again. The cheap green-ratio test short-circuits before any inference, so out-of-play frames cost 0.4 ms instead of 42 ms. `benchmark.py` reports detector calls per frame as a guard: it must stay at 1.00.

ByteTrack only advances its internal clock inside `update()`. When a play phase ends the pipeline calls `tracker.reset()`, otherwise skipping a cut lets stale ids re-associate on resume.

**Evaluation** (`src/evaluation`) is the measurement layer the rest exists to serve. `mot.py` handles MOTChallenge IO, `metrics.py` scores, `soccernet.py` reads SoccerNet sequences, `cli.py` exposes both CLIs.

## Invariants worth not breaking

Metrics are never reimplemented. HOTA, DetA and AssA come from TrackEval, MOTA, IDF1, precision and recall from motmetrics. One IoU routine in `metrics.py` feeds both so they cannot disagree about what counts as a match.

MOT frame numbers are 1-based and local to the evaluated sequence. They are not `Frame.frame_id`, which indexes the source video and survives seeking. Ground truth must be labelled over the same window the pipeline ran on.

TrackEval sizes its accumulators by the largest track id value, so `_dense_ids` compacts ids before HOTA runs. Scores are unaffected but a ByteTrack id of 5000 would allocate 200 MB and divide zero by zero on the unused rows.

Combining sequences is not averaging their scores. `evaluate_many` pools detection counts and weights association by true positives, following TrackEval, so long clips count for more. `eval-dataset` prints both so the gap stays visible.

SoccerNet ground truth annotates the ball as an ordinary tracklet, around 6% of rows and a few pixels wide. `Sequence.ground_truth()` drops it by role via `gameinfo.ini`. Referees and goalkeepers stay, being detectable people.

`tests/test_bytetrack_tracker.py` must not reload `bytetrack_tracker` against a fake `supervision` without restoring it. A reloaded module keeps its own reference to the stub, which monkeypatch cannot undo, and every later test in the session then builds a tracker returning canned boxes. An autouse fixture restores it and a guard test asserts the restoration happened.

`outputs/` is fully gitignored. Prediction files land there and are large.

## Project context

This is layer 1 of a five-layer research project:

* Vision: video to players, ball, pitch, tracking
* Representation: tracking to continuous tactical states as latent embeddings
* Discovery: states to unsupervised tactical patterns
* World model: dynamics to simulated game sequences
* Decision support: simulation to interpretable strategic recommendations

World model and decision support are the last layers, not the next ones. The immediate consumer of anything built here is representation, which needs continuous tactical state: player positions over time in pitch coordinates, with team identity, and trajectories that do not break. That makes track fragmentation costlier than a tracking metric suggests, since an embedding built on a trajectory that splits in two is learning from a fiction. It also makes pitch calibration a prerequisite rather than a nice-to-have, because image coordinates are not a tactical state.

Choices are expected to be justified by a delta measured through the evaluation harness, not by preference.

Visualisation is a requirement of every phase, not a final deliverable. Each phase should produce something to look at, not only numbers: overlays showing what the model actually saw, diagnostics showing why a metric came out as it did, and pitch-space views once calibration exists. A result that cannot be shown is not finished.

Current baseline on the full SoccerNet test split is recorded in `results/soccernet-test.md`: HOTA 44.17 against a published 42.38, with detection recall identified as the dominant error term. Perfect boxes move HOTA to 82.31, so detection is worth attacking before tracker parameters.

`supervision.ByteTrack` is deprecated since supervision 0.28.0 and removed in 0.30.0, replaced by the separate `trackers` package. Migrating would invalidate the recorded baseline, so rerun `results/soccernet-test.md` if it happens.

`config.yaml` and `README.md` are empty, and `pyyaml` is an unused dependency.
