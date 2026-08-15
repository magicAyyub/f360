import yaml
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path("config.yaml")

DEFAULTS: Dict[str, Any] = {
    "video": None,
    "start_time": 0.0,
    "end_time": None,
    "threshold": 0.5,
    "device": None,
    "weights": "models/transnetv2/transnetv2-pytorch-weights.pth",
    "shots_path": "outputs/shots.json",
    "clip_dir": "outputs/clips",
    "labels_path": "outputs/labels.json",
}


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Read the settings file, falling back to the built-in defaults."""
    settings = dict(DEFAULTS)
    if not path.is_file():
        return settings

    stored = yaml.safe_load(path.read_text()) or {}
    unknown = set(stored) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings in {path}: {', '.join(sorted(unknown))}")

    # Une cle laissee vide vaut None et retombe donc sur le defaut.
    settings.update({key: value for key, value in stored.items() if value is not None})
    return settings


def resolve(path: Path = CONFIG_PATH, **overrides: Any) -> Dict[str, Any]:
    """Settings from the file, overridden by the options actually passed on the command line."""
    settings = load_config(path)
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return settings
