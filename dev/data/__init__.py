'''
This module provides functions for loading modeling data.
'''

import gzip as gz
from cPickle import load
import Image
from numpy import array, ones

def format_input(imgArray):
<<<<<<< HEAD
    imgArray = numpy.subtract(imgArray,127.5)
    return imgArray / 127.5

=======
    imgArray = imgArray / 127.5
    return imgArray - ones(len(imgArray)) 
>>>>>>> gabor_update
    

def load_img(imgpath, dims):
    img = Image.open(imgpath).resize(dims).getdata()
    img.convert('L')
    return array(img)


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
    


