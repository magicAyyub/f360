# f360

Detects camera cuts in a broadcast football video and cuts the video into one clip per shot.

A televised match is a single video stream, but the director keeps switching between cameras.
Everything here is about finding those switch points precisely, nothing else: no player, ball or
field detection yet.

## Setup

```bash
uv sync
```

You also need ffmpeg (`brew install ffmpeg`) and the TransNetV2 weights, which are not in the repo:

```bash
mkdir -p models/transnetv2
curl -L -o models/transnetv2/transnetv2-pytorch-weights.pth \
  https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights/resolve/main/transnetv2-pytorch-weights.pth
```

## Running it

Every command below is prefixed with `uv run`, which runs it inside the project environment without
having to activate anything. If you already have the venv activated, drop the prefix.

Detection. Edit the video path and time window at the top of `main.py`, then:

```bash
uv run python main.py
```

It writes `outputs/shots.json`. Expect roughly 30 seconds of compute per minute of video.

Cutting. Once the JSON exists:

```bash
uv run clips export     # one mp4 per shot in outputs/clips
uv run clips clean      # delete them
```

The `clips` command comes from `pyproject.toml` and is created when the project is installed. If it
is not found, run `uv sync` again, or fall back to `uv run python -m src.clips export`.

## Reading the output

```json
{
  "index": 4,
  "start_frame": 18793,
  "end_frame": 18984,
  "start_time": 319.481,
  "end_time": 322.745,
  "duration": 3.264
}
```

Frame bounds are inclusive, `end_time` is exclusive. That is deliberate: `end_time` is the exact
instant the cut happens, so `-ss start_time -to end_time` gives back exactly the frames of the shot,
with none lost or duplicated.

Consecutive shots do not touch: there is always at least one frame between `end_frame` of one and
`start_frame` of the next. Those are the transition frames, and they belong to no shot on purpose.
A TV mixer rarely cuts cleanly from one camera to the next, it blends them over a frame or two, so
those frames show both shots at once and would pollute whichever clip they landed in.

## Things worth knowing

Frames must be consecutive. `VideoReader` accepts a `stride`, but feeding a strided stream to the
detector fabricates cuts everywhere, since the model reads a transition as an abrupt change between
neighbouring frames. Detection always runs at `stride=1`.

Boundaries are accurate to within one frame. The model flags the frames it considers part of the
transition and we drop them, but it can miss a frame that is only faintly blended, in which case a
clip keeps a slightly contaminated frame at its edge. Lowering `threshold` on `ShotDetector` widens
what counts as a transition, at the cost of trimming clean frames.

Timestamps assume a constant frame rate, since they are computed as `frame_id / fps`. On a variable
frame rate file they would drift, and the frame numbers would be the only reliable bounds.

Clips are re-encoded, not stream-copied. A stream copy would be near instant but would snap every
cut to the nearest keyframe, which throws away the precision above.

## Layout

- `src/reader.py` reads a video, optionally within a time window, with seeking rather than decoding
  and discarding everything before the start.
- `src/shot_detector.py` runs TransNetV2 and turns per-frame transition probabilities into shots.
- `src/clips.py` cuts and cleans the clips.
- `src/vendor/` holds the TransNetV2 model definition, copied unchanged from the official repo.
