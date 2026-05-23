from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import statistics
import time

from src.pipeline import LidarPerceptionPipeline, iter_lidar_frames


def main():
    pipeline = LidarPerceptionPipeline()
    frame_paths = iter_lidar_frames()
    durations = []

    for file_path in frame_paths:
        start = time.perf_counter()
        result = pipeline.process_frame(file_path)
        durations.append(time.perf_counter() - start)
        print(
            f"{file_path}: points={result['points'].shape[0]} "
            f"clusters={len(result['clusters'])} tracked={len(result['tracked_objects'])} "
            f"time={durations[-1]:.4f}s"
        )

    if durations:
        print("--- Benchmark Summary ---")
        print(f"frames={len(durations)}")
        print(f"mean={statistics.mean(durations):.4f}s")
        print(f"median={statistics.median(durations):.4f}s")
        print(f"max={max(durations):.4f}s")


if __name__ == "__main__":
    main()
