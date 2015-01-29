#split lena
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

lena = scipy.misc.lena()
mat = lena#np.arange(100.0).reshape(10, 10)
patch_w = 74
patch_h = 74
def patchify(img, patch_shape):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
        
    #shape = (3,3,x,y)
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    print "strides",strides
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

x = patchify(mat,(patch_h,patch_w))

patches = np.ascontiguousarray(x)
print patches.shape
#which ones do you want?
#if you're not doing it evenly, you'll need to do
#something like the last line...
y = []
for i in range(len(patches)):
    for j in range(len(patches[i])):
        if not(i%69) and not(j%69):
            y.append(patches[i][j])
#        if j==len(patches[i])-1:
#            print i,j,patches[i][j]

#for n in range(len(y)):
#    plt.figure(n)
#    plt.imshow(y[n],cmap='gray')

#plt.show()

def get_img_array():
    return y



    



