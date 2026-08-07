from typing import Optional

import numpy as np
import typer

from src.detection import YoloDetector
from src.evaluation import save_mot, tracks_to_mot
from src.evaluation.soccernet import load_sequence
from src.tracking import ByteTrackTracker
from src.video import ImageSequenceReader

app = typer.Typer(help="Run detection and tracking over a MOT-format image sequence")

PERSON_CLASS = 0


def run(
    sequence_dir: str,
    output_tracks: str,
    conf: float = 0.25,
    limit: Optional[int] = None,
) -> None:
    """Track a SoccerNet sequence and write predictions in MOTChallenge format.

    No play-phase filtering here: these clips are already continuous play, and
    dropping frames would break alignment with the ground truth.
    """
    sequence = load_sequence(sequence_dir)
    reader = ImageSequenceReader(sequence.image_dir, fps=sequence.fps)

    print(f'{sequence.name}: {len(reader)} frames, {sequence.width}x{sequence.height} @ {sequence.fps:g} fps')
    print(f'tracklet roles: {sequence.role_counts()}')

    detector = YoloDetector(conf=conf, classes=[PERSON_CLASS])
    tracker = ByteTrackTracker(frame_rate=sequence.fps)

    rows = []
    for frame_number, frame in enumerate(reader, start=1):
        if limit is not None and frame_number > limit:
            break

        tracks = tracker.update(detector.predict(frame.image))
        rows.append(tracks_to_mot(frame_number, tracks))

        if frame_number % 100 == 0:
            print(f'  {frame_number} frames')

    stacked = np.vstack(rows) if rows else np.zeros((0, 7))
    save_mot(stacked, output_tracks)
    print(f'Wrote {len(stacked)} track rows to: {output_tracks}')


@app.command()
def cli(
    sequence_dir: str = typer.Argument(..., help='Sequence directory containing seqinfo.ini and img1'),
    output_tracks: str = typer.Option(..., help='Path to write MOTChallenge predictions'),
    conf: float = typer.Option(0.25, help='Detector confidence threshold'),
    limit: Optional[int] = typer.Option(None, help='Stop after this many frames'),
) -> None:
    run(sequence_dir=sequence_dir, output_tracks=output_tracks, conf=conf, limit=limit)


def run_cli() -> None:
    """Entry point for packaging: starts the Typer CLI."""
    app()


if __name__ == '__main__':
    app()
