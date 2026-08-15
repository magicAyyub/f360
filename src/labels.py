import json
import subprocess
import sys
import typer
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

from src.config import resolve

# Touche: etiquette, couleur. Un ralenti reste un ralenti quel que soit le cadrage.
LABELS = {
    "w": ("live_wide", "green"),
    "r": ("replay", "yellow"),
    "c": ("close_up", "cyan"),
    "a": ("crowd", "magenta"),
    "b": ("bench", "blue"),
    "g": ("graphics", "red"),
}

REPLAY, SKIP, UNDO, QUIT = " ", "s", "u", "q"

console = Console()


def load_labels(labels_path: Path) -> Dict[int, str]:
    """Labels already recorded, so a session can be stopped and resumed."""
    if not labels_path.is_file():
        return {}

    stored = json.loads(labels_path.read_text())
    return {entry["index"]: entry["label"] for entry in stored["labels"]}


def save_labels(labels: Dict[int, str], shots: List[dict], labels_path: Path, video_path: str) -> None:
    """Write the labels next to the frame bounds they belong to, so the file stands alone."""
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    entries = [
        {
            "index": shot["index"],
            "start_frame": shot["start_frame"],
            "end_frame": shot["end_frame"],
            "label": labels[shot["index"]],
        }
        for shot in shots
        if shot["index"] in labels
    ]
    labels_path.write_text(json.dumps({"video": video_path, "labels": entries}, indent=2))


def label(
    shots_path: Optional[Path] = typer.Option(None, help="Path to the shots JSON file."),
    clip_dir: Optional[Path] = typer.Option(None, help="Where the clips were written."),
    labels_path: Optional[Path] = typer.Option(None, help="Where the labels are written."),
    config: Path = typer.Option("config.yaml", help="Settings file."),
):
    """Watch each clip and record what kind of shot it holds."""
    if sys.platform == "win32":
        typer.secho("label needs a Unix terminal, it does not run on Windows yet", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        settings = resolve(config, shots_path=shots_path, clip_dir=clip_dir, labels_path=labels_path)
        stored = json.loads(Path(settings["shots_path"]).read_text())
    except ValueError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError:
        typer.secho(f"{settings['shots_path']} not found, run `detect` first", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    shots = stored["shots"]
    clips = Path(settings["clip_dir"])
    destination = Path(settings["labels_path"])

    labels = load_labels(destination)
    pending = [shot for shot in shots if shot["index"] not in labels]

    if not pending:
        console.print(f"all {len(shots)} shots already labelled in {destination}")
        return

    if not (clips / f"shot_{pending[0]['index']:03d}.mp4").is_file():
        typer.secho(f"no clip in {clips}, run `export` first", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    _legend(len(labels), len(shots))

    cursor, playing = 0, -1
    while cursor < len(pending):
        shot = pending[cursor]

        clip = clips / f"shot_{shot['index']:03d}.mp4"

        if playing != cursor:
            _play(clip)
            console.print(
                f"  [dim]{shot['index']:>3}/{len(shots)}[/]  "
                f"{_clock(shot['start_time'])} [dim]→[/] {_clock(shot['end_time'])}  "
                f"[dim]{shot['duration']:5.1f}s[/]  ",
                end="",
            )
            playing = cursor

        key = _read_key().lower()

        if key == QUIT:
            console.print("[dim]quit[/]")
            break

        if key == REPLAY:
            _play(clip)
            continue

        if key == UNDO:
            console.print("[dim]undo[/]")
            cursor = max(0, cursor - 1)
            labels.pop(pending[cursor]["index"], None)
            save_labels(labels, shots, destination, stored["video"])
            playing = -1
            continue

        if key == SKIP:
            console.print("[dim]skipped[/]")
            cursor += 1
            playing = -1
            continue

        if key in LABELS:
            name, colour = LABELS[key]
            console.print(f"[{colour}]{name}[/]")
            labels[shot["index"]] = name
            save_labels(labels, shots, destination, stored["video"])
            cursor += 1
            playing = -1

    _close_player()
    _summary(labels, len(shots), destination)


def _legend(done: int, total: int) -> None:
    keys = "   ".join(f"[{colour}]{key}[/] {name}" for key, (name, colour) in LABELS.items())
    console.print(f"\n  {keys}")
    console.print("  [dim]space replay   s skip   u undo   q quit[/]")
    console.print(f"  [dim]{done} of {total} already labelled[/]\n")


def _summary(labels: Dict[int, str], total: int, destination: Path) -> None:
    console.print(f"\n  [dim]{len(labels)} of {total} labelled, written to {destination}[/]")
    for name, colour in LABELS.values():
        count = sum(1 for value in labels.values() if value == name)
        if count:
            console.print(f"    [{colour}]{name:<10}[/] {count}")


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _play(clip: Path) -> None:
    if sys.platform != "darwin":
        subprocess.Popen(["xdg-open", str(clip)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    # Une seule fenetre a la fois: on ferme le document precedent avant d'ouvrir
    # le suivant. Sans "activate", le clavier reste au terminal.
    done = _osascript(
        f'close every document saving no\n'
        f'open POSIX file "{clip.resolve()}"\n'
        f'play front document'
    )
    if not done:
        subprocess.Popen(["open", "-g", str(clip)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _close_player() -> None:
    if sys.platform == "darwin":
        _osascript("close every document saving no")


def _osascript(body: str) -> bool:
    script = f'tell application "QuickTime Player"\n{body}\nend tell'
    result = subprocess.run(
        ["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return result.returncode == 0


def _read_key() -> str:
    """One keypress, without waiting for Enter. Needs a Unix terminal."""
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)

    # En mode brut, Ctrl-C arrive comme un caractere ordinaire.
    return QUIT if key == "\x03" else key
