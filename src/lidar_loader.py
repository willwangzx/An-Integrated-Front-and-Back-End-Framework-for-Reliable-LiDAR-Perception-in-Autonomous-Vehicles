import laspy
import numpy as np


def load_las(file_path):
    las = laspy.read(file_path)

    points = np.column_stack((las.x, las.y, las.z)).astype(np.float32, copy=False)

    if hasattr(las, "intensity"):
        raw_intensity = np.asarray(las.intensity)
        intensity = raw_intensity.astype(np.float32, copy=False)

        if np.issubdtype(raw_intensity.dtype, np.integer):
            dtype_max = np.iinfo(raw_intensity.dtype).max
            if dtype_max > 0:
                intensity = intensity / dtype_max
    else:
        intensity = np.ones(points.shape[0], dtype=np.float32)

    return points, intensity.astype(np.float32, copy=False)
