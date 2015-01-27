from models.big_model import *

T = array([j.flatten()/norm(j.flatten()) for j in mk_bgbrs(5000, (28,28), 28.0/16)])
IMG = load_img('./data/lena_512x512.png', (28,28)).flatten()
print norm(IMG.flatten())
from numpy import dot
t = dot(T.T, T)
##for i in range(len(T[0])):
##    print t[i][i]

#from numpy import reshape
#import matplotlib.pyplot as plt
#re = dot(T, IMG)
#re = dot(T.T, re)
#re = reshape(re, (28, 28))
#plt.imshow(re, cmap='gray')
#plt.show()
