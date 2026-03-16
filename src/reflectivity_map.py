def build_reflectivity(voxels,intensity):

    reflectivity={}

    for v,i in zip(voxels,intensity):

        reflectivity[tuple(v)]=i

    return reflectivity