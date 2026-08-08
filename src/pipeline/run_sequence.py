from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

from src.detection import YoloDetector
from src.evaluation import save_mot, tracks_to_mot
from src.evaluation.soccernet import load_sequence
from src.tracking import ByteTrackTracker
from src.video import ImageSequenceReader

app = typer.Typer(help="Run detection and tracking over MOT-format image sequences")

PERSON_CLASS = 0


def find_sequences(path: Path) -> List[Path]:
    """Accept either one sequence directory or a dataset root holding many."""
    if (path / 'seqinfo.ini').exists():
        return [path]

    found = sorted(child for child in path.iterdir() if (child / 'seqinfo.ini').exists())
    if not found:
        raise FileNotFoundError(f"no sequence with seqinfo.ini under {path}")
    return found


def track_sequence(sequence_dir: Path, detector: YoloDetector, limit: Optional[int] = None) -> np.ndarray:
    """Track one sequence and return its MOT rows.

    No play-phase filtering: these clips are continuous play already, and dropping
    frames would break alignment with the ground truth.
    """
    sequence = load_sequence(sequence_dir)
    reader = ImageSequenceReader(sequence.image_dir, fps=sequence.fps)

    # Un tracker neuf par séquence: les clips sont indépendants
    tracker = ByteTrackTracker(frame_rate=sequence.fps)

    rows = []
    for frame_number, frame in enumerate(reader, start=1):
        if limit is not None and frame_number > limit:
            break
        rows.append(tracks_to_mot(frame_number, tracker.update(detector.predict(frame.image))))

    return np.vstack(rows) if rows else np.zeros((0, 7))


def run(
    path: str,
    output_dir: str,
    conf: float = 0.25,
    limit: Optional[int] = None,
) -> None:
    sequences = find_sequences(Path(path))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # Un seul chargement de modèle pour tout le lot
    detector = YoloDetector(conf=conf, classes=[PERSON_CLASS])
    print(f'{len(sequences)} sequence(s) to process, writing to {destination}')

    for index, sequence_dir in enumerate(sequences, start=1):
        rows = track_sequence(sequence_dir, detector, limit=limit)
        output = destination / f'{sequence_dir.name}.txt'
        save_mot(rows, output)
        print(f'  [{index}/{len(sequences)}] {sequence_dir.name}: {len(rows)} rows -> {output}')


@app.command()
def cli(
    path: str = typer.Argument(..., help='A sequence directory, or a dataset root containing several'),
    output_dir: str = typer.Option(..., help='Directory to write one MOTChallenge file per sequence'),
    conf: float = typer.Option(0.25, help='Detector confidence threshold'),
    limit: Optional[int] = typer.Option(None, help='Stop each sequence after this many frames'),
) -> None:
    run(path=path, output_dir=output_dir, conf=conf, limit=limit)


def run_cli() -> None:
    """Entry point for packaging: starts the Typer CLI."""
    app()


if __name__ == '__main__':
    app()
