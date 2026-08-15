import json
import shutil
import subprocess
import typer
from pathlib import Path
from rich.progress import Progress
from typing import Callable, List, Optional

from src.config import DEFAULTS, resolve

CLIP_DIR = Path(DEFAULTS["clip_dir"])
CLIP_PATTERN = "shot_*.mp4"


def export_clips(
    shots_path: Path,
    output_dir: Path = CLIP_DIR,
    video_path: Optional[str] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> List[Path]:
    """Cut one clip per shot described in a shots JSON file.

    `progress` is called with 1 after each clip.
    """
    _require_ffmpeg()

    data = json.loads(Path(shots_path).read_text())
    source = video_path or data.get("video")
    if not source:
        raise ValueError(f"no source video in {shots_path}, pass video_path explicitly")

    output_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    for shot in data["shots"]:
        destination = output_dir / f"shot_{shot['index']:03d}.mp4"
        _cut(source, shot["start_time"], shot["end_time"] - shot["start_time"], destination)
        clips.append(destination)

        if progress is not None:
            progress(1)

    return clips


def count_shots(shots_path: Path) -> int:
    """How many shots a JSON file holds, useful to size a progress bar."""
    return json.loads(shots_path.read_text())["shot_count"]


def clear_clips(output_dir: Path = CLIP_DIR) -> int:
    """Remove previously generated clips and return how many were deleted."""
    if not output_dir.is_dir():
        return 0

    clips = sorted(output_dir.glob(CLIP_PATTERN))
    for clip in clips:
        clip.unlink()

    return len(clips)


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH, install it first (brew install ffmpeg)")


def _cut(source: str, start: float, duration: float, destination: Path) -> None:
    # Re-encodage volontaire: une copie de flux recalerait la coupe sur la
    # keyframe la plus proche et ruinerait la precision des frontieres.
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}",
        "-i", source,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        str(destination),
    ]
    subprocess.run(command, check=True)


def export(
    shots_path: Optional[Path] = typer.Option(None, help="Path to the shots JSON file."),
    clip_dir: Optional[Path] = typer.Option(None, help="Where the clips are written."),
    video: Optional[str] = typer.Option(None, help="Override the source video referenced in the JSON file."),
    config: Path = typer.Option("config.yaml", help="Settings file."),
):
    """Write one clip per detected shot."""
    try:
        settings = resolve(config, shots_path=shots_path, clip_dir=clip_dir)
    except ValueError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    shots_file = Path(settings["shots_path"])
    try:
        total = count_shots(shots_file)
    except FileNotFoundError:
        typer.secho(f"{shots_file} not found, run `detect` first", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # La video retenue lors de la detection est inscrite dans le JSON: elle prime
        # sur la config, seule une demande explicite en ligne de commande la remplace.
        with Progress() as progress:
            cutting = progress.add_task("Cutting clips", total=total)
            clips = export_clips(
                shots_file,
                output_dir=Path(settings["clip_dir"]),
                video_path=video,
                progress=lambda step: progress.advance(cutting, step),
            )
    except (RuntimeError, ValueError) as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Wrote {len(clips)} clips to {settings['clip_dir']}")


def clean(
    clip_dir: Optional[Path] = typer.Option(None, help="Directory to clean."),
    config: Path = typer.Option("config.yaml", help="Settings file."),
):
    """Delete the clips previously generated in the output directory."""
    settings = resolve(config, clip_dir=clip_dir)
    directory = Path(settings["clip_dir"])
    typer.echo(f"Removed {clear_clips(directory)} clips from {directory}")
