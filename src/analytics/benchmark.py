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
from src.tracking import ByteTrackTracker, SamSegmenter


def benchmark(
    video_path: str = 'data/PSG_vs_Bayern_Munchen.mp4',
    start_time: float = 315.0,
    num_frames: int = 60,
    sam_model: str = 'models/sam2_t.pt'
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
    detector = YoloDetector()
    tracker = ByteTrackTracker()
    play_detector = PlayPhaseDetector()
    sam_segmenter = SamSegmenter(model_path=sam_model)
    init_time = time.time() - init_start
    console.print(f"Component Initialization Time: [bold yellow]{init_time:.2f}[/bold yellow] seconds")

    # Load video reader
    reader = VideoReader(video_path)
    reader = TimeWindowFilter(reader, start_time=start_time, end_time=start_time + 10.0)

    # Metrics storage
    times = {
        "read": [],
        "play_detect": [],
        "yolo": [],
        "bytetrack": [],
        "sam2": [],
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

    for frame in reader:
        if frame_count >= num_frames:
            break
        
        t_frame_start = time.time()
        
        # 1. Read / Decode Frame
        t_read = time.time() - t_frame_start
        times["read"].append(t_read)

        # 2. Play Detection Heuristics
        t_play_start = time.time()
        in_play, play_metrics = play_detector.is_in_play(frame.image, detector)
        t_play = time.time() - t_play_start
        times["play_detect"].append(t_play)

        t_yolo = 0.0
        t_bytetrack = 0.0
        t_sam2 = 0.0

        if in_play:
            in_play_count += 1
            # 3. YOLO Detection
            t_yolo_start = time.time()
            pred = detector.predict(frame.image)
            t_yolo = time.time() - t_yolo_start
            
            # 4. ByteTrack Tracking
            t_bytetrack_start = time.time()
            tracks = tracker.update(pred)
            t_bytetrack = time.time() - t_bytetrack_start
            
            total_players_detected += len(tracks)

            # 5. SAM2 Segmentation
            if len(tracks) > 0:
                t_sam2_start = time.time()
                bboxes = np.array([t.box for t in tracks])
                _ = sam_segmenter.segment(frame.image, bboxes)
                t_sam2 = time.time() - t_sam2_start

        times["yolo"].append(t_yolo)
        times["bytetrack"].append(t_bytetrack)
        times["sam2"].append(t_sam2)
        times["total_frame"].append(time.time() - t_frame_start)
        
        frame_count += 1

    if frame_count == 0:
        console.print("[bold red]No frames were processed.[/bold red]")
        return

    # Calculate statistics
    avg_fps = 1.0 / np.mean(times["total_frame"])
    avg_total = np.mean(times["total_frame"]) * 1000
    avg_read = np.mean(times["read"]) * 1000
    avg_play = np.mean(times["play_detect"]) * 1000
    avg_yolo = np.mean([t for t in times["yolo"] if t > 0]) * 1000 if in_play_count > 0 else 0.0
    avg_byte = np.mean([t for t in times["bytetrack"] if t > 0]) * 1000 if in_play_count > 0 else 0.0
    avg_sam = np.mean([t for t in times["sam2"] if t > 0]) * 1000 if in_play_count > 0 else 0.0

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
    breakdown_table.add_row("Play Detection (Heuristics)", f"{avg_play:.1f} ms")
    breakdown_table.add_row("YOLO Object Detection", f"{avg_yolo:.1f} ms" if in_play_count > 0 else "N/A")
    breakdown_table.add_row("ByteTrack Tracker", f"{avg_byte:.1f} ms" if in_play_count > 0 else "N/A")
    breakdown_table.add_row("SAM2 Player Segmentation", f"{avg_sam:.1f} ms" if in_play_count > 0 else "N/A")

    console.print(breakdown_table)


if __name__ == "__main__":
    benchmark()
