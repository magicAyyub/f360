import typer

from src.clips import clean, export
from src.detect import detect
from src.labels import label

app = typer.Typer(help="Detect the camera cuts of a broadcast video and cut it into shots.")

app.command()(detect)
app.command()(export)
app.command()(label)
app.command()(clean)


if __name__ == "__main__":
    app()
