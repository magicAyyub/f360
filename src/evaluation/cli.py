from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .metrics import evaluate, evaluate_many
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


dataset_app = typer.Typer(help="Score a whole dataset of sequences")


@dataset_app.command()
def dataset_cli(
    dataset_dir: str = typer.Argument(..., help='Root holding one directory per sequence'),
    predictions_dir: str = typer.Argument(..., help='Directory of <sequence>.txt prediction files'),
    iou_threshold: float = typer.Option(0.5, help='IoU required for a match in the CLEAR metrics'),
    exclude_roles: str = typer.Option('ball', help='Roles to drop from ground truth'),
    show_sequences: bool = typer.Option(False, help='Also print a row per sequence'),
) -> None:
    """Score every sequence that has a prediction file, and combine them."""
    roles = tuple(r.strip() for r in exclude_roles.split(',') if r.strip())
    console = Console()

    pairs = {}
    for sequence_dir in sorted(Path(dataset_dir).iterdir()):
        prediction = Path(predictions_dir) / f'{sequence_dir.name}.txt'
        if not (sequence_dir / 'seqinfo.ini').exists() or not prediction.exists():
            continue
        pairs[sequence_dir.name] = (
            load_sequence(sequence_dir).ground_truth(exclude_roles=roles),
            load_mot(prediction),
        )

    if not pairs:
        raise typer.BadParameter(f"no sequence in {dataset_dir} has a prediction in {predictions_dir}")

    per_sequence, overall = evaluate_many(pairs, iou_threshold=iou_threshold)

    if show_sequences:
        table = Table(title=f"Per sequence ({len(per_sequence)} clips)", header_style="bold cyan")
        for column in ('Sequence', 'HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'Recall', 'IDsw'):
            table.add_column(column, justify='right' if column != 'Sequence' else 'left')
        for name in sorted(per_sequence, key=lambda n: per_sequence[n].hota):
            m = per_sequence[name]
            table.add_row(name, f"{m.hota:.4f}", f"{m.deta:.4f}", f"{m.assa:.4f}",
                          f"{m.mota:.4f}", f"{m.idf1:.4f}", f"{m.recall:.4f}", str(m.id_switches))
        console.print(table)

    scores = overall.as_dict()
    hotas = [m.hota for m in per_sequence.values()]
    summary = Table(title=f"Combined over {len(per_sequence)} sequences (IoU >= {iou_threshold})",
                    header_style="bold magenta")
    summary.add_column("Metric", style="dim", width=24)
    summary.add_column("Value", justify="right", width=12)

    for section, fields in GROUPS:
        summary.add_section()
        summary.add_row(f"[bold cyan]{section}[/bold cyan]", "")
        for key, label in fields:
            value = scores[key]
            summary.add_row(f"  {label}", f"{value:.4f}" if isinstance(value, float) else str(value))

    summary.add_section()
    summary.add_row("[bold cyan]Spread[/bold cyan]", "")
    summary.add_row("  HOTA min", f"{min(hotas):.4f}")
    summary.add_row("  HOTA max", f"{max(hotas):.4f}")
    summary.add_row("  HOTA mean of clips", f"{float(np.mean(hotas)):.4f}")

    console.print(summary)
    console.print("[dim]combined is not the mean of clips: counts are pooled, so long clips weigh more[/dim]")


def run_cli() -> None:
    """Entry point for packaging: starts the Typer CLI."""
    app()


def run_dataset_cli() -> None:
    """Entry point for packaging: starts the dataset-level Typer CLI."""
    dataset_app()


if __name__ == '__main__':
    app()
