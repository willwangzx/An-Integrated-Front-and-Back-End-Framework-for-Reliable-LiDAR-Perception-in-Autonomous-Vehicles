import laspy
import numpy as np


def load_las(file_path):

    las = laspy.read(file_path)

    # 读取坐标
    x = las.x
    y = las.y
    z = las.z

    points = np.vstack((x, y, z)).T

    # 读取反射强度
    if hasattr(las, "intensity"):
        intensity = las.intensity.astype(np.float32)
        intensity = intensity / (intensity.max() + 1e-6)
    else:
        intensity = np.ones(points.shape[0])

    return points, intensity