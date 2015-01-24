'''Tools for displaying data.'''

import os
import Image
from numpy import reshape
import numpy
from matplotlib.pyplot import imshow, savefig, axis, figure, close, imsave


def img_from_vector(vector, dims):
    v = vector.reshape(dims).astype('uint8')
    return Image.fromarray(v, 'L')


def mk_plt_imgs(path, data):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(data.data)):
        name = path+'%03d.png' % (i+1)
        img = reshape(data.data[i], data.dims, 'F')
        imsave(name, img.T, cmap='gray')
        #fig = figure(figsize=(data.dims[0],data.dims[1]), frameon=False)
        #imshow(img.T, cmap='gray', interpolation='nearest')
        #axis('off')
        #fig.axes.get_xaxis().set_visible(False)
        #fig.axes.get_yaxis().set_visible(False)
        #fig.tight_layout()
        #savefig(name, bbox_inches='tight', pad_inches=0, dpi=1)
        #close('all')
        print 'Saved img %d of %d' % (i+1, len(data.data))
    print 'Done.'

    
def mk_imgs(path, data):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(data.data)):
        name = path+'%03d.png' % (i+1)
        img_from_vector(data.data[i], data.dims).save(name)
        print 'Saved img %d of %d' % (i+1, len(data.data))
    avg = sum(data.data)/len(data.data)
    img_from_vector(avg, data.dims).save(path+'avg.png')
    print 'Saved average of images.'
    print 'Done.'
        

    
