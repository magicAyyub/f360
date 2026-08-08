import numpy as np
import pytest

from src.evaluation import load_sequence


SEQ_INFO = """[Sequence]
name=SNMOT-999
imDir=img1
frameRate=25
seqLength=3
imWidth=1920
imHeight=1080
imExt=.jpg
"""

GAME_INFO = """[Sequence]
name=SNMOT-999
num_tracklets=4
trackletID_1= player team left;4
trackletID_2= player team right;25
trackletID_3= referee;main
trackletID_4= ball;1
"""

# frame, id, x, y, w, h, conf, puis trois colonnes inutilisées comme en MOT20
GROUND_TRUTH = """1,1,10,10,50,120,1,-1,-1,-1
1,2,80,10,50,120,1,-1,-1,-1
1,3,150,10,50,120,1,-1,-1,-1
1,4,300,300,5,5,1,-1,-1,-1
2,1,12,10,50,120,1,-1,-1,-1
2,4,305,300,5,5,1,-1,-1,-1
"""


@pytest.fixture
def sequence_dir(tmp_path):
    directory = tmp_path / 'SNMOT-999'
    (directory / 'gt').mkdir(parents=True)
    (directory / 'img1').mkdir()
    (directory / 'seqinfo.ini').write_text(SEQ_INFO)
    (directory / 'gameinfo.ini').write_text(GAME_INFO)
    (directory / 'gt' / 'gt.txt').write_text(GROUND_TRUTH)
    return directory


def test_reads_sequence_metadata(sequence_dir):
    sequence = load_sequence(sequence_dir)

    assert sequence.name == 'SNMOT-999'
    assert sequence.fps == 25.0
    assert sequence.length == 3
    assert (sequence.width, sequence.height) == (1920, 1080)
    assert sequence.image_dir == sequence_dir / 'img1'


def test_parses_tracklet_roles(sequence_dir):
    sequence = load_sequence(sequence_dir)

    assert sequence.roles == {
        1: 'player team left',
        2: 'player team right',
        3: 'referee',
        4: 'ball',
    }
    assert sequence.role_counts()['referee'] == 1


def test_ball_excluded_by_default(sequence_dir):
    sequence = load_sequence(sequence_dir)

    kept = sequence.ground_truth()
    assert len(kept) == 4
    assert 4 not in kept[:, 1].astype(int)

    # Les arbitres et gardiens restent: ce sont des personnes détectables
    assert set(kept[:, 1].astype(int)) == {1, 2, 3}


def test_excluding_nothing_keeps_every_row(sequence_dir):
    assert len(load_sequence(sequence_dir).ground_truth(exclude_roles=())) == 6


def test_excluding_several_roles(sequence_dir):
    kept = load_sequence(sequence_dir).ground_truth(exclude_roles=('ball', 'referee'))
    assert set(kept[:, 1].astype(int)) == {1, 2}


def test_ten_column_rows_are_truncated_to_seven(sequence_dir):
    rows = load_sequence(sequence_dir).ground_truth()
    assert rows.shape[1] == 7
    assert np.allclose(rows[0], [1, 1, 10, 10, 50, 120, 1])


def test_missing_game_info_leaves_roles_empty(sequence_dir):
    (sequence_dir / 'gameinfo.ini').unlink()
    sequence = load_sequence(sequence_dir)

    assert sequence.roles == {}
    # Sans rôles connus, rien ne peut être filtré
    assert len(sequence.ground_truth()) == 6


def test_find_sequences_accepts_a_single_sequence(sequence_dir):
    from src.pipeline.run_sequence import find_sequences

    assert find_sequences(sequence_dir) == [sequence_dir]


def test_find_sequences_accepts_a_dataset_root(sequence_dir):
    from src.pipeline.run_sequence import find_sequences

    root = sequence_dir.parent
    (root / 'not-a-sequence').mkdir()

    assert find_sequences(root) == [sequence_dir]


def test_find_sequences_rejects_empty_root(tmp_path):
    from src.pipeline.run_sequence import find_sequences

    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError):
        find_sequences(tmp_path / 'empty')
