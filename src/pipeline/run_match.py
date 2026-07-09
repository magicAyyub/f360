from typing import Optional, Tuple
import typer
import cv2
import numpy as np
from pathlib import Path

from src.video import VideoReader, FrameSampler, TimeWindowFilter
from src.analytics import PlayPhaseDetector
from src.detection import YoloDetector
from src.tracking import ByteTrackTracker

app = typer.Typer(help="Football analytics pipeline CLI")


def run(
    video_path: str = 'data/PSG_vs_Bayern_Munchen.mp4',
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    stride: int = 1,
    resize: Optional[Tuple[int, int]] = None,
    output_video: Optional[str] = None,
    display: bool = False,
    detect_play: bool = False,
    only_in_play: bool = False,
) -> None:
    print(f'Running pipeline on: {video_path} (start={start_time} end={end_time} stride={stride} resize={resize})')
    reader = VideoReader(video_path)

    # Setup video writer if output requested
    video_writer = None
    frame_width, frame_height = None, None
    if output_video:
        Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    if start_time is not None or end_time is not None:
        reader = TimeWindowFilter(reader, start_time=start_time or 0.0, end_time=end_time)
    if stride != 1 or resize is not None:
        reader = FrameSampler(reader, stride=stride, resize=resize)

    detector = YoloDetector()
    tracker = ByteTrackTracker()
    play_detector = PlayPhaseDetector() if detect_play else None

    for frame in reader:
        in_play = True
        play_metrics = {}

        if play_detector:
            in_play, play_metrics = play_detector.is_in_play(frame.image, detector)

        if not in_play and only_in_play:
            continue

        if in_play:
            pred = detector.predict(frame.image)
            tracks = tracker.update(pred)
        else:
            tracks = []

        # Draw bounding boxes and play phase on the frame
        frame_vis = frame.image.copy()

        if play_detector:
            status_text = f"PLAYING (Green: {play_metrics.get('green_ratio', 0.0):.2f})" if in_play else f"OUT OF PLAY ({play_metrics.get('reason', '')})"
            color = (0, 255, 0) if in_play else (0, 0, 255)
            cv2.putText(frame_vis, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if in_play:
            for t in tracks:
                x1, y1, x2, y2 = t.box.astype(int)
                # Draw rectangle and track ID
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Track {t.track_id} ({t.score:.2f})"
                cv2.putText(frame_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                print(f"Track {t.track_id}: box={x1,y1,x2,y2} score={t.score:.2f} class={t.class_id}")

        # Initialize video writer on first frame
        if output_video and video_writer is None:
            frame_height, frame_width = frame_vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, 30.0, (frame_width, frame_height))

        # Write to output video
        if video_writer is not None:
            video_writer.write(frame_vis)

        # Display frame if requested (small delay so it's visible)
        if display:
            cv2.imshow('Pipeline Debug', frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    if video_writer is not None:
        video_writer.release()
        print(f"Output video saved to: {output_video}")

    if display:
        cv2.destroyAllWindows()


@app.command()
def cli(
    video_path: str = typer.Option('data/PSG_vs_Bayern_Munchen.mp4', help='Path to input video'),
    start_time: Optional[float] = typer.Option(None, help='Start time in seconds'),
    end_time: Optional[float] = typer.Option(None, help='End time in seconds'),
    stride: int = typer.Option(1, help='Frame stride (sampling)'),
    resize: Optional[str] = typer.Option(None, help='Resize as WIDTHxHEIGHT, e.g. 1280x720'),
    output_video: Optional[str] = typer.Option(None, help='Path to save output video with tracks drawn'),
    display: bool = typer.Option(False, help='Display frames in a window during processing'),
    detect_play: bool = typer.Option(False, help='Enable play phase detection'),
    only_in_play: bool = typer.Option(False, help='Only process and output frames that are in play'),
) -> None:
    """Run the match processing pipeline."""
    resize_tuple: Optional[Tuple[int, int]] = None
    if resize:
        try:
            w, h = resize.lower().split('x')
            resize_tuple = (int(w), int(h))
        except Exception:
            raise typer.BadParameter("resize must be like 1280x720")

    run(video_path=video_path, start_time=start_time, end_time=end_time, stride=stride, resize=resize_tuple,
        output_video=output_video, display=display, detect_play=detect_play, only_in_play=only_in_play)


def run_cli() -> None:
    """Entry point for packaging: starts the Typer CLI."""
    app()


if __name__ == '__main__':
    app()

