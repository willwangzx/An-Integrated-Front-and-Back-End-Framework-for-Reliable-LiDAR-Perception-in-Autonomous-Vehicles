from collections import defaultdict
import numpy as np

def fuse_maps(maps,use_ewma=False,ewma_alpha=0.6):

    fusion=defaultdict(list)

    for m in maps:

        for k,v in m.items():

            fusion[k].append(v)

    fused={}
    stability={}
    n_frames=max(len(maps),1)

    for k,v in fusion.items():

        values=np.array(v,dtype=np.float32)
        if use_ewma and values.size>0:
            ewma=values[0]
            for val in values[1:]:
                ewma=ewma_alpha*val+(1-ewma_alpha)*ewma
            fused[k]=float(ewma)
        else:
            fused[k]=float(np.mean(values))
        stability[k]=float(values.size/n_frames)

    return fused,stability
