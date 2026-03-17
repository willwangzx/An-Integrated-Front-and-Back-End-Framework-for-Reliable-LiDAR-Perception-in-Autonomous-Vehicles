import numpy as np
from collections import defaultdict

def voxelize(points,intensity,voxel_size):

    if points.size == 0:
        empty_idx = np.empty((0, 3), dtype=np.int32)
        empty_float = np.empty((0,), dtype=np.float32)
        empty_counts = np.empty((0,), dtype=np.int32)
        return empty_idx, empty_float, empty_counts

    voxel_dict = defaultdict(list)

    for p,i in zip(points,intensity):

        idx = tuple(np.floor(p/voxel_size).astype(int))

        voxel_dict[idx].append(i)

    voxels=[]
    intensities=[]
    counts=[]

    for k,v in voxel_dict.items():

        voxels.append(k)
        intensities.append(np.mean(v))
        counts.append(len(v))

    voxels=np.asarray(voxels, dtype=np.int32)
    intensities=np.asarray(intensities, dtype=np.float32)
    counts=np.asarray(counts, dtype=np.int32)

    return voxels,intensities,counts
