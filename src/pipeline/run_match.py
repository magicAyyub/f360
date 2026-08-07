from typing import Optional, Tuple
import typer
import cv2
from pathlib import Path

from src.video import VideoReader, FrameSampler, TimeWindowFilter
from src.analytics import PlayPhaseDetector
from src.detection import YoloDetector
from src.tracking import ByteTrackTracker

app = typer.Typer(help="Football analytics pipeline CLI")

# COCO person: seul classe pertinente ici, sinon ByteTrack suit aussi
# les "sports ball", "frisbee", "tv" que YOLO sort sur du foot télévisé
PERSON_CLASS = 0


def run(
    video_path: str = 'data/PSG_vs_Bayern_Munchen.mp4',
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    stride: int = 1,
    resize: Optional[Tuple[int, int]] = None,
    output_video: Optional[str] = None,
    display: bool = False,
    detect_play: bool = True,
    only_in_play: bool = False,
) -> None:
    print(f'Running pipeline on: {video_path} (start={start_time} end={end_time} stride={stride} resize={resize})')

    # Le seek évite de décoder puis jeter toutes les frames avant start_time
    reader = VideoReader(video_path, start_time=start_time or 0.0)

    # Setup video writer if output requested
    video_writer = None
    if output_video:
        Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    if start_time is not None or end_time is not None:
        reader = TimeWindowFilter(reader, start_time=start_time or 0.0, end_time=end_time)
    if stride != 1 or resize is not None:
        reader = FrameSampler(reader, stride=stride, resize=resize)

    # Cadence réelle des frames émises, stride compris: pilote ByteTrack et l'écriture
    output_fps = reader.fps
    print(f'Effective frame rate: {output_fps:.2f} fps')

    detector = YoloDetector(classes=[PERSON_CLASS])
    tracker = ByteTrackTracker(frame_rate=output_fps)
    play_detector = PlayPhaseDetector() if detect_play else None

    was_in_play = False

    for frame in reader:
        play_metrics = {}
        prediction = None
        in_play = True

        if play_detector:
            phase = play_detector.evaluate(frame.image, detector)
            in_play, play_metrics, prediction = phase.in_play, phase.metrics, phase.prediction

        if in_play:
            if prediction is None:
                prediction = detector.predict(frame.image)
            tracks = tracker.update(prediction)
        else:
            # ByteTrack n'avance son compteur interne que sur update(), donc sauter
            # une coupure réassocierait des IDs périmés à la reprise
            if was_in_play:
                tracker.reset()
            tracks = []

        was_in_play = in_play

        if not in_play and only_in_play:
            continue

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
            video_writer = cv2.VideoWriter(output_video, fourcc, output_fps, (frame_width, frame_height))

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
    detect_play: bool = typer.Option(True, help='Enable play phase detection'),
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

