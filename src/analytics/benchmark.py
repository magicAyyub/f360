import time
import os
import psutil
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from src.video import VideoReader, TimeWindowFilter
from src.analytics import PlayPhaseDetector
from src.detection import YoloDetector
from src.detection.base import Detector, Prediction
from src.tracking import ByteTrackTracker


class TimedDetector(Detector):
    """Detector proxy recording inference time and call count for one frame."""

    def __init__(self, inner: Detector) -> None:
        self.inner = inner
        self.elapsed = 0.0
        self.calls = 0

    def reset(self) -> None:
        self.elapsed = 0.0
        self.calls = 0

    def predict(self, frame) -> Prediction:
        start = time.time()
        prediction = self.inner.predict(frame)
        self.elapsed += time.time() - start
        self.calls += 1
        return prediction


def benchmark(
    video_path: str = 'data/PSG_vs_Bayern_Munchen.mp4',
    start_time: float = 315.0,
    num_frames: int = 60,
) -> None:
    """Benchmark performance metrics and component latency for the pipeline."""
    console = Console()
    console.print(f"[bold green]Starting pipeline benchmark on:[/bold green] {video_path}")
    console.print(f"Start Time: {start_time}s | Frames to evaluate: {num_frames}\n")

    if not os.path.exists(video_path):
        console.print(f"[bold red]Error:[/bold red] Video file not found at {video_path}")
        return

    # Initialize components
    init_start = time.time()
    detector = YoloDetector(classes=[PlayPhaseDetector.PERSON_CLASS])
    play_detector = PlayPhaseDetector()
    init_time = time.time() - init_start
    console.print(f"Component Initialization Time: [bold yellow]{init_time:.2f}[/bold yellow] seconds")

    # Load video reader
    reader = VideoReader(video_path, start_time=start_time)
    reader = TimeWindowFilter(reader, start_time=start_time, end_time=start_time + 10.0)
    tracker = ByteTrackTracker(frame_rate=reader.fps)

    # Metrics storage
    times = {
        "read": [],
        "green_ratio": [],
        "yolo": [],
        "bytetrack": [],
        "total_frame": []
    }

    frame_count = 0
    in_play_count = 0
    total_players_detected = 0

    # Device verification
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = "CUDA"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "MPS (Apple Silicon)"

    console.print(f"Running on Device: [bold cyan]{device_name}[/bold cyan]")
    console.print("Processing frames...\n")

    # Le détecteur est instrumenté pour mesurer YOLO là où il est réellement appelé,
    # et pour compter les appels par frame (garde-fou contre l'inférence en double)
    timed_detector = TimedDetector(detector)
    detector_calls = []
    was_in_play = False

    for frame in reader:
        if frame_count >= num_frames:
            break

        t_frame_start = time.time()

        # 1. Read / Decode Frame
        t_read = time.time() - t_frame_start
        times["read"].append(t_read)

        # 2. Green ratio heuristic, measured on its own
        t_green_start = time.time()
        play_detector.compute_green_ratio(frame.image)
        times["green_ratio"].append(time.time() - t_green_start)

        # 3. Play phase decision, which calls YOLO at most once
        timed_detector.reset()
        phase = play_detector.evaluate(frame.image, timed_detector)
        in_play = phase.in_play

        t_bytetrack = 0.0

        if in_play:
            in_play_count += 1
            prediction = phase.prediction
            if prediction is None:
                prediction = timed_detector.predict(frame.image)

            # 4. ByteTrack Tracking
            t_bytetrack_start = time.time()
            tracks = tracker.update(prediction)
            t_bytetrack = time.time() - t_bytetrack_start

            total_players_detected += len(tracks)
        elif was_in_play:
            tracker.reset()

        was_in_play = in_play

        detector_calls.append(timed_detector.calls)
        times["yolo"].append(timed_detector.elapsed)
        times["bytetrack"].append(t_bytetrack)
        times["total_frame"].append(time.time() - t_frame_start)

        frame_count += 1

    if frame_count == 0:
        console.print("[bold red]No frames were processed.[/bold red]")
        return

    # Calculate statistics
    avg_fps = 1.0 / np.mean(times["total_frame"])
    avg_total = np.mean(times["total_frame"]) * 1000
    avg_read = np.mean(times["read"]) * 1000
    avg_green = np.mean(times["green_ratio"]) * 1000
    avg_yolo = np.mean([t for t in times["yolo"] if t > 0]) * 1000 if in_play_count > 0 else 0.0
    avg_byte = np.mean([t for t in times["bytetrack"] if t > 0]) * 1000 if in_play_count > 0 else 0.0

    # Memory usage
    process = psutil.Process(os.getpid())
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)

    # Output results using Rich Table
    results_table = Table(title="Pipeline Benchmark Results", show_header=True, header_style="bold magenta")
    results_table.add_column("Metric", style="dim", width=30)
    results_table.add_column("Value", justify="right", width=25)

    results_table.add_row("Total Frames Evaluated", f"{frame_count}")
    results_table.add_row("Frames In Play", f"{in_play_count} ({in_play_count/frame_count*100:.1f}%)")
    results_table.add_row("Average Pipeline Speed", f"{avg_fps:.2f} FPS")
    results_table.add_row("Average Frame Latency", f"{avg_total:.1f} ms")
    results_table.add_row("Detector Calls / Frame", f"{np.mean(detector_calls):.2f} (max {max(detector_calls)})")
    results_table.add_row("Memory Consumption (RAM)", f"{ram_usage_mb:.1f} MB")
    
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        results_table.add_row("Peak VRAM Allocated", f"{vram:.1f} MB")

    console.print(results_table)
    console.print()

    breakdown_table = Table(title="Component Latency Breakdown", show_header=True, header_style="bold cyan")
    breakdown_table.add_column("Component", style="dim", width=30)
    breakdown_table.add_column("Average Latency", justify="right", width=25)

    breakdown_table.add_row("Frame Decode (Reader)", f"{avg_read:.1f} ms")
    breakdown_table.add_row("Green Ratio Heuristic", f"{avg_green:.1f} ms")
    breakdown_table.add_row("YOLO Object Detection", f"{avg_yolo:.1f} ms" if in_play_count > 0 else "N/A")
    breakdown_table.add_row("ByteTrack Tracker", f"{avg_byte:.1f} ms" if in_play_count > 0 else "N/A")

    console.print(breakdown_table)


if __name__ == "__main__":
    benchmark()
