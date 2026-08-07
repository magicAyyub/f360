import cv2 as cv

from src.video.reader import FrameSampler, TimeWindowFilter, VideoReader

video_path = 'data/PSG_vs_Bayern_Munchen.mp4'
print('reading the video file...')

reader = VideoReader(video_path, start_time=300.0)
window = TimeWindowFilter(reader, start_time=300.0, end_time=360.0)
sampler = FrameSampler(window, stride=2, resize=(1280, 720))

delay = max(1, int(1000 / sampler.fps))

for frame in sampler:
    cv.imshow('Video', frame.image)
    if cv.waitKey(delay) & 0xFF == 27:
        break

cv.destroyAllWindows()
