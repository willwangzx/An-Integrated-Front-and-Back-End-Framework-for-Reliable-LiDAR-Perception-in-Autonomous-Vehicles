import numpy as np

def generate_bev(points,resolution=0.2):

    x=points[:,0]
    y=points[:,1]

    xmin,xmax=x.min(),x.max()
    ymin,ymax=y.min(),y.max()

    width=int((xmax-xmin)/resolution)+1
    height=int((ymax-ymin)/resolution)+1

    bev=np.zeros((width,height))

    xi=((x-xmin)/resolution).astype(int)
    yi=((y-ymin)/resolution).astype(int)

    bev[xi,yi]+=1

    return bev