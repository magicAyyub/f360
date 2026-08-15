import shutil


def require() -> None:
    """Fail early, and with a readable message, when ffmpeg is not installed."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH, install it first (brew install ffmpeg)")
