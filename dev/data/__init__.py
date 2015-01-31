'''
This module provides functions for loading modeling data.
'''

import gzip as gz
from cPickle import load, dump
import Image
from numpy import array, ones, subtract, meshgrid, linspace, cos, sin, pi
from numpy.random import uniform
from numpy.linalg import norm
import random
from matplotlib.pyplot import imsave, imshow
import matplotlib.pyplot as plt
import os

def mk_gratings(canvas_size,filename):
    gratings = []
    lambds = [2**3,2**2,2,1,2**-1,2**-2,2**-3]
    thetas = [0,2*pi/3,4*pi/3]    
    for theta in thetas:
        for lambd in lambds:
            #uniform(0,2*pi)
            psi = uniform(0,2*pi)
        
            g = grating(canvas_size,lambd,theta,psi)
            gratings.append([[canvas_size,lambd,theta,psi],g])
            #gratings.append(g)
            #imshow(gratings[(canvas_size,lambd,theta,psi)], cmap='gray')
            #plt.show()
        
        

    #if not os.path.exists(filename):
    #    os.makedirs(filename)
    with gz.open(filename, 'wb') as f:
        dump(gratings, f)

    
            
        

def grating(canvas_size, lambd, theta, psi):
    '''Returns a single gabor filter.
    '''
    x = linspace(-1, 1, canvas_size)
    y = linspace(-1, 1, canvas_size)
    X, Y = meshgrid(x, y)


    cosTheta = cos(theta)
    sinTheta = sin(theta)
    xTheta = X * cosTheta  + Y * sinTheta
    yTheta = -X * sinTheta + Y * cosTheta
    
    cosed = cos((2 * pi * xTheta / lambd) + psi)
    return cosed
    
def load_img(imgpath, dims):
    img = Image.open(imgpath).resize(dims).getdata()
    img.convert('L')
    img = subtract(array(img).flatten(), 127.5)
    return img/norm(img)


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
    


