from src.reader import VideoReader
from src.shot_detector import ShotDetector, save_shots

VIDEO_PATH = 'data/PSG_vs_Bayern_Munchen.mp4'
OUTPUT_PATH = 'outputs/shots.json'
START_TIME = 300.0
END_TIME = 360.0

reader = VideoReader(VIDEO_PATH, start_time=START_TIME, end_time=END_TIME)
detector = ShotDetector()

print(f'detecting shot boundaries on [{START_TIME:.0f}s, {END_TIME:.0f}s] using {detector.device}...')
shots = detector.detect(reader)

save_shots(shots, OUTPUT_PATH, video_path=VIDEO_PATH)
print(f'{len(shots)} shots written to {OUTPUT_PATH}')
