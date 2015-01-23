'''Tools for displaying data.'''

import os
import Image
from numpy import reshape
import numpy
from matplotlib.pyplot import imshow, savefig, clf



def mk_plt_imgs(path,data):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(data.data)):
        name = path+'%03d.png' % (i+1)
        img = reshape(data.data[i], data.dims, 'F')
        imshow(img.T, cmap='gray')
        savefig(name)
        clf()

         
def img_from_vector(vector, dims):
    v = vector.reshape(dims).astype('uint8')
    return Image.fromarray(v, 'L')

def convert_output(imgArray):
    imgArray = imgArray * 127.5
    return numpy.around(numpy.add(imgArray,127.5),decimals=0)
    
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
        

    
