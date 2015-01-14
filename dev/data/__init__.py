'''
This module provides functions for loading modeling data.
'''

import gzip as gz
from cPickle import load


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
    

