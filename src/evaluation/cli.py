from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .metrics import evaluate
from .mot import load_mot
from .soccernet import load_sequence

app = typer.Typer(help="Score tracker output against ground truth")

# Regroupées par ce qu'elles mesurent, pas par ordre alphabétique
GROUPS = (
    ("Overall", (('hota', 'HOTA'), ('mota', 'MOTA'), ('idf1', 'IDF1'))),
    ("Detection", (('deta', 'DetA'), ('precision', 'Precision'), ('recall', 'Recall'),
                   ('motp', 'MOTP (IoU)'))),
    ("Association", (('assa', 'AssA'), ('id_switches', 'ID switches'))),
    ("Counts", (('false_positives', 'False positives'), ('misses', 'Misses'),
                ('num_gt_boxes', 'Ground truth boxes'), ('num_pred_boxes', 'Predicted boxes'),
                ('num_frames', 'Frames'))),
)


def _load_ground_truth(path: str, exclude_roles: str) -> tuple[np.ndarray, str]:
    """Accept either a MOT file or a SoccerNet sequence directory."""
    if not Path(path).is_dir():
        return load_mot(path), Path(path).name

    sequence = load_sequence(path)
    roles = tuple(r.strip() for r in exclude_roles.split(',') if r.strip())
    rows = sequence.ground_truth(exclude_roles=roles)
    excluded = {tid for tid, role in sequence.roles.items() if role in roles}
    return rows, f"{sequence.name} (excluded {len(excluded)} tracklets: {', '.join(roles) or 'none'})"


@app.command()
def cli(
    ground_truth: str = typer.Argument(..., help='MOTChallenge file, or a SoccerNet sequence directory'),
    predictions: str = typer.Argument(..., help='MOTChallenge predictions file'),
    iou_threshold: float = typer.Option(0.5, help='IoU required for a match in the CLEAR metrics'),
    exclude_roles: str = typer.Option('ball', help='Roles to drop when ground truth is a sequence directory'),
) -> None:
    """Compare predictions against ground truth and print the tracking metrics."""
    gt, label = _load_ground_truth(ground_truth, exclude_roles)
    result = evaluate(gt, load_mot(predictions), iou_threshold=iou_threshold)
    scores = result.as_dict()

    Console().print(f"[dim]ground truth:[/dim] {label}")
    table = Table(title=f"Tracking metrics (IoU >= {iou_threshold})", header_style="bold magenta")
    table.add_column("Metric", style="dim", width=24)
    table.add_column("Value", justify="right", width=12)

    for section, fields in GROUPS:
        table.add_section()
        table.add_row(f"[bold cyan]{section}[/bold cyan]", "")
        for key, label in fields:
            value = scores[key]
            table.add_row(f"  {label}", f"{value:.4f}" if isinstance(value, float) else str(value))

    Console().print(table)


def run_cli() -> None:
    """Entry point for packaging: starts the Typer CLI."""
    app()


if __name__ == '__main__':
    app()
