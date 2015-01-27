'''
This module provides functions for loading modeling data.
'''

import gzip as gz
from cPickle import load
import Image
from numpy import array, ones, subtract
from numpy.linalg import norm

    
def load_img(imgpath, dims):
    img = Image.open(imgpath).resize(dims).getdata()
    img.convert('L')
    img = array(img)
    
    img = subtract(img,127.5)
    img = img.flatten()
    return img / norm(img)
    
   # return (array(img)- 127.5*ones(dims[0]*dims[1])) 


def load_data(filename):
    return load(gz.open(filename))


def load_mini_mnist(option=None):
    mmnist = load(gz.open('./data/mini_mnist.pkl.gz', 'rb'))
    if option == 'train':
        return mmnist[0]
    elif option == 'valid':
        return mmnist[1]
    elif option == 'test':
        return mmnist[2]
    else:
        return mmnist
    


