import numpy as np
from collections import defaultdict

def voxelize(points,intensity,voxel_size):

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

    voxels=np.array(voxels)*voxel_size
    intensities=np.array(intensities)
    counts=np.array(counts)

    return voxels,intensities,counts
