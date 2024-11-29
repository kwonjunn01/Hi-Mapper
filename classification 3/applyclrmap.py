from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy as np


v = 1
#colors = [white,빨강(채도50%),빨강(채도100%)]
# rgb로 변경 -> mcl.hsv_to_rgb(h,s,v)

colors1 = [
    mcl.hsv_to_rgb((25/360,0.5,v)),
    mcl.hsv_to_rgb((25/360,1,v)),
    mcl.hsv_to_rgb((252/360,0.5,v)),
    mcl.hsv_to_rgb((252/360,1,v)),
    mcl.hsv_to_rgb((139/360,0.5,v)),
    mcl.hsv_to_rgb((139/360,1,v)),
    mcl.hsv_to_rgb((296/360,0.5,v)),
    mcl.hsv_to_rgb((296/360,1,v))
]


cmap = LinearSegmentedColormap.from_list('my_cmap',colors,gamma=2)
def imtoclr(img, L):
    coef = int(256/L)
    img = (img+1)*coef
    plt.imsave(img,cmap=cmap,interpolation='none')
    

