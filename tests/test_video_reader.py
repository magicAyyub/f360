import cv2
import numpy as np
import pytest

from src.video import FrameSampler, TimeWindowFilter, VideoReader


SOURCE_FPS = 50.0
NUM_FRAMES = 100


@pytest.fixture
def video_file(tmp_path):
    """Small synthetic clip at a non-30 fps, to catch hardcoded frame rates."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), SOURCE_FPS, (64, 48))
    for i in range(NUM_FRAMES):
        frame = np.full((48, 64, 3), i * 2, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


def test_reader_reports_source_fps(video_file):
    assert VideoReader(video_file).fps == pytest.approx(SOURCE_FPS)


def test_sampler_fps_accounts_for_stride(video_file):
    reader = VideoReader(video_file)
    assert FrameSampler(reader, stride=2).fps == pytest.approx(SOURCE_FPS / 2)
    assert TimeWindowFilter(reader, start_time=0.5).fps == pytest.approx(SOURCE_FPS)


def test_timestamps_use_source_fps(video_file):
    frames = list(VideoReader(video_file))
    assert len(frames) == NUM_FRAMES
    assert frames[int(SOURCE_FPS)].timestamp == pytest.approx(1.0)


def test_seek_preserves_frame_ids(video_file):
    start = 1.0
    seeking = list(VideoReader(video_file, start_time=start))
    decoded = [f for f in VideoReader(video_file) if f.timestamp >= start]

    # Le seek peut démarrer un peu avant sur une keyframe, jamais après
    assert seeking[0].frame_id <= decoded[0].frame_id
    assert seeking[-1].frame_id == decoded[-1].frame_id

    # Les ids restent alignés sur la source, donc les timestamps aussi
    by_id = {f.frame_id: f.timestamp for f in seeking}
    for frame in decoded:
        assert by_id[frame.frame_id] == pytest.approx(frame.timestamp)


def test_window_after_seek_is_exact(video_file):
    reader = VideoReader(video_file, start_time=1.0)
    frames = list(TimeWindowFilter(reader, start_time=1.0, end_time=1.5))

    assert frames, "window should not be empty"
    assert frames[0].timestamp >= 1.0
    assert frames[-1].timestamp <= 1.5


@pytest.fixture
def image_sequence(tmp_path):
    directory = tmp_path / 'img1'
    directory.mkdir()
    for i in range(1, 6):
        cv2.imwrite(str(directory / f'{i:06d}.jpg'), np.full((48, 64, 3), i * 10, dtype=np.uint8))
    return directory


def test_image_sequence_reader(image_sequence):
    from src.video import ImageSequenceReader

    reader = ImageSequenceReader(image_sequence, fps=25.0)
    assert len(reader) == 5
    assert reader.fps == 25.0

    frames = list(reader)
    assert [f.frame_id for f in frames] == [0, 1, 2, 3, 4]
    assert frames[4].timestamp == pytest.approx(4 / 25.0)
    # Le tri lexicographique doit suivre l'ordre temporel des noms zéro-remplis
    assert frames[0].image.mean() < frames[4].image.mean()


def test_image_sequence_composes_with_sampler(image_sequence):
    from src.video import FrameSampler, ImageSequenceReader

    sampler = FrameSampler(ImageSequenceReader(image_sequence, fps=25.0), stride=2)
    assert sampler.fps == pytest.approx(12.5)
    assert [f.frame_id for f in sampler] == [0, 2, 4]


def test_image_sequence_rejects_empty_dir(tmp_path):
    from src.video import ImageSequenceReader

    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError):
        ImageSequenceReader(tmp_path / 'empty', fps=25.0)
