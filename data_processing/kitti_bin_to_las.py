import numpy as np
import pdal
import json
import os

def kitti_bin_to_las(
    bin_path,
    las_path,
    scale=0.001
):
    # 1. Load KITTI bin
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError("Invalid KITTI bin file: size not divisible by 4")
    pts = raw.reshape(-1, 4)

    # 2. Structured numpy array
    arr = np.zeros(
        pts.shape[0],
        dtype=[
            ("X", np.float32),
            ("Y", np.float32),
            ("Z", np.float32),
            ("Intensity", np.uint16),
        ],
    )

    arr["X"] = -pts[:, 1]  # 右
    arr["Y"] = pts[:, 0]  # 前
    arr["Z"] = pts[:, 2]  # 上
    arr["Intensity"] = pts[:, 3]

    # 3. PDAL pipeline
    pipeline_json = {
        "pipeline": [
            #{"type": "readers.memory"},
            {
                "type": "writers.las",
                "filename": las_path,
                "scale_x": scale,
                "scale_y": scale,
                "scale_z": scale,
                "extra_dims": "Intensity=float",
            },
        ]
    }

    pipeline = pdal.Pipeline(
        json.dumps(pipeline_json),
        arrays=[arr], 
    )

    pipeline.execute()
    #print(f"[OK] {bin_path} → {las_path}")

if __name__ == "__main__":
    import glob
    import os

    bin_dir = "C:/Users/willw/PycharmProjects/Lidar/data/velodyne_points/data"
    las_dir = "C:/Users/willw/PycharmProjects/Lidar/data/velodyne_points/las"

    os.makedirs(las_dir, exist_ok=True)

    for bin_path in glob.glob(os.path.join(bin_dir, "*.bin")):
        name = os.path.splitext(os.path.basename(bin_path))[0]
        las_path = os.path.join(las_dir, name + ".las")
        kitti_bin_to_las(bin_path, las_path)

