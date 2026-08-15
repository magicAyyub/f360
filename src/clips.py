import json
import shutil
import subprocess
import typer
from pathlib import Path
from typing import List, Optional

DEFAULT_SHOTS = Path("outputs/shots.json")
CLIP_DIR = Path("outputs/clips")
CLIP_PATTERN = "shot_*.mp4"

app = typer.Typer(help="Cut a video into clips from a shots JSON file.")


def export_clips(shots_path: Path, output_dir: Path = CLIP_DIR, video_path: Optional[str] = None) -> List[Path]:
    """Cut one clip per shot described in a shots JSON file."""
    _require_ffmpeg()

    data = json.loads(Path(shots_path).read_text())
    source = video_path or data.get("video")
    if not source:
        raise ValueError(f"no source video in {shots_path}, pass video_path explicitly")

    output_dir.mkdir(parents=True, exist_ok=True)

    shots = data["shots"]
    clips = []
    for position, shot in enumerate(shots, start=1):
        destination = output_dir / f"shot_{shot['index']:03d}.mp4"
        _cut(source, shot["start_time"], shot["end_time"] - shot["start_time"], destination)
        clips.append(destination)
        print(f"\rCutting clips: {position}/{len(shots)}", end="", flush=True)

    print()
    return clips


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


@app.command()
def export(
    shots: Path = typer.Argument(DEFAULT_SHOTS, exists=True, help="Path to the shots JSON file."),
    output_dir: Path = typer.Option(CLIP_DIR, help="Where the clips are written."),
    video: Optional[str] = typer.Option(None, help="Override the source video referenced in the JSON file."),
):
    """Write one clip per detected shot."""
    try:
        clips = export_clips(shots, output_dir=output_dir, video_path=video)
    except (RuntimeError, ValueError) as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Wrote {len(clips)} clips to {output_dir}")


@app.command()
def clean(output_dir: Path = typer.Option(CLIP_DIR, help="Directory to clean.")):
    """Delete the clips previously generated in the output directory."""
    typer.echo(f"Removed {clear_clips(output_dir)} clips from {output_dir}")


if __name__ == "__main__":
    app()
