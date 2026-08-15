import typer
from pathlib import Path
from rich.progress import Progress
from typing import Optional

from src.config import resolve
from src.reader import VideoReader
from src.shot_detector import ShotDetector, save_shots

def detect(
    video: Optional[str] = typer.Option(None, help="Video to analyse."),
    start_time: Optional[float] = typer.Option(None, help="Start of the analysed window, in seconds."),
    end_time: Optional[float] = typer.Option(None, help="End of the analysed window, in seconds."),
    threshold: Optional[float] = typer.Option(None, help="Transition probability above which a frame is a cut."),
    device: Optional[str] = typer.Option(None, help="cpu, mps or cuda."),
    shots_path: Optional[Path] = typer.Option(None, help="Where the JSON file is written."),
    config: Path = typer.Option("config.yaml", help="Settings file."),
):
    """Locate every camera cut and describe the resulting shots."""
    try:
        settings = resolve(
            config,
            video=video,
            start_time=start_time,
            end_time=end_time,
            threshold=threshold,
            device=device,
            shots_path=shots_path,
        )
    except ValueError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if not settings["video"]:
        typer.secho("no video to analyse, set it in the settings file or pass --video", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Frames consecutives obligatoires: un stride fabriquerait des transitions.
    reader = VideoReader(
        settings["video"],
        start_time=settings["start_time"],
        end_time=settings["end_time"],
    )
    detector = ShotDetector(
        weights_path=settings["weights"],
        threshold=settings["threshold"],
        device=settings["device"],
    )

    window = f"{settings['start_time']:.0f}s" + (f" to {settings['end_time']:.0f}s" if settings["end_time"] else " to the end")
    typer.echo(f"Analysing {window} of {settings['video']} on {detector.device}")

    with Progress() as progress:
        frames = progress.track(reader, total=reader.frame_count, description="Reading frames")
        images, timeline = detector.collect(frames)

        detecting = progress.add_task("Detecting cuts", total=len(images))
        probabilities = detector.predict(images, progress=lambda step: progress.advance(detecting, step))

    shots = detector.to_shots(probabilities, timeline)
    save_shots(shots, settings["shots_path"], video_path=settings["video"])
    typer.echo(f"Wrote {len(shots)} shots to {settings['shots_path']}")
