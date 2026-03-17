from src.config import *
from src.pipeline import LidarPerceptionPipeline, iter_lidar_frames
from src.visualization import visualize_points,visualize_clusters
pipeline = LidarPerceptionPipeline()

files = iter_lidar_frames()

for idx,f in enumerate(files):

    result = pipeline.process_frame(f)
    points = result["points"]
    clusters = result["clusters"]
    object_features = result["object_features"]

    print(
        f"Frame {idx}: points={points.shape[0]} "
        f"clusters={len(clusters)} objects={len(object_features)}"
    )

    if (
        ENABLE_VISUALIZATION
        and points.shape[0] > 0
        and (VISUALIZE_EVERY_N==0 or idx%VISUALIZE_EVERY_N==0)
    ):
        visualize_clusters(clusters)
        visualize_points(points)

print("Perception pipeline finished")


