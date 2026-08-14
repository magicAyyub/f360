import cv2 as cv

from src.reader import FrameSampler, TimeWindowFilter, VideoReader

video_path = 'data/PSG_vs_Bayern_Munchen.mp4'
print('reading the video file...')

reader = VideoReader(video_path)
window = TimeWindowFilter(reader, start_time=300.0, end_time=360.0)
sampler = FrameSampler(window, stride=2, resize=(1280, 720))

for frame in sampler:
    cv.imshow('Video', frame.image)
    if cv.waitKey(int(1000 / 30)) & 0xFF == 27:
        break

cv.destroyAllWindows()