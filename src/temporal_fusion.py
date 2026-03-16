from collections import defaultdict
import numpy as np

def fuse_maps(maps):

    fusion=defaultdict(list)

    for m in maps:

        for k,v in m.items():

            fusion[k].append(v)

    fused={}

    for k,v in fusion.items():

        fused[k]=np.mean(v)

    return fused