import cv2


class VideoReader:
    """Iterable wrapper around a video file, yielding frames one by one."""

    def __init__(self, video_path: str):
        self.video_path = video_path

    def __iter__(self):
        # Re-open the capture on each iteration so the reader can be looped over more than once.
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_id = 0
        while True:
            success, frame = cap.read()
            if not success:  # end of stream or read error
                break

            yield Frame(
                frame_id=frame_id,
                timestamp=frame_id / fps,
                image=frame,
            )
            frame_id += 1

        cap.release()