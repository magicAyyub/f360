from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence as SequenceType

import numpy as np

from .mot import COLUMNS, load_mot


BALL_ROLE = 'ball'


@dataclass(frozen=True)
class Sequence:
    """One SoccerNet tracking clip: MOT layout plus a role for each tracklet."""

    name: str
    directory: Path
    fps: float
    length: int
    width: int
    height: int
    image_dir: Path
    roles: Dict[int, str]

    def ground_truth(self, exclude_roles: SequenceType[str] = (BALL_ROLE,)) -> np.ndarray:
        """Ground truth rows, dropping tracklets whose role is excluded.

        The ball is excluded by default: it is annotated as a tracklet but is a
        few pixels wide and no person detector can produce it, so keeping it only
        depresses recall for a reason unrelated to tracking quality.
        """
        rows = load_mot(self.directory / 'gt' / 'gt.txt')
        if not exclude_roles:
            return rows

        excluded = {tid for tid, role in self.roles.items() if role in exclude_roles}
        if not excluded:
            return rows

        keep = ~np.isin(rows[:, 1].astype(int), list(excluded))
        return rows[keep]

    def role_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for role in self.roles.values():
            counts[role] = counts.get(role, 0) + 1
        return counts


def _read_ini(path: Path) -> configparser.SectionProxy:
    parser = configparser.ConfigParser()
    parser.read(path)
    return parser['Sequence']


def load_sequence(directory: str | Path) -> Sequence:
    """Read a SoccerNet tracking sequence directory."""
    directory = Path(directory)
    info = _read_ini(directory / 'seqinfo.ini')

    roles: Dict[int, str] = {}
    game_info_path = directory / 'gameinfo.ini'
    if game_info_path.exists():
        game = _read_ini(game_info_path)
        for key, value in game.items():
            # configparser abaisse la casse des clés: trackletID_7 devient trackletid_7
            if not key.startswith('trackletid_'):
                continue
            # Valeur du type " player team left;4", le rôle est avant le point-virgule
            roles[int(key.split('_')[1])] = value.split(';')[0].strip()

    return Sequence(
        name=info['name'],
        directory=directory,
        fps=float(info['framerate']),
        length=int(info['seqlength']),
        width=int(info['imwidth']),
        height=int(info['imheight']),
        image_dir=directory / info['imdir'],
        roles=roles,
    )
