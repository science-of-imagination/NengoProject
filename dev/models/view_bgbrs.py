from models.big_model import *
dims = (40,40)
T = array([j.flatten()/norm(j.flatten()) for j in mk_bgbrs(10000,
                                                           dims,
                                                           dims[0]*4,
                                                           x_off=lambda:uniform(-0.5,0.5))])
IMG = load_img('./data/lena_512x512.png', dims)
print norm(IMG.flatten())
from numpy import dot
#t = dot(T.T, T)
#for i in range(len(T[0])):
#    print t[i][i]

from numpy import reshape
import matplotlib.pyplot as plt
re = dot(T, IMG)
re = dot(T.T, re)
re = reshape(re, dims)
plt.imshow((2/(184.0+263))*re, cmap='gray')
plt.show()
