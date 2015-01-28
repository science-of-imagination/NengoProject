'''Utility to collect data on simulations'''

import os
import gzip as gz
from cPickle import dump
from time import strftime
from numpy import mean, dot, subtract, sqrt
    

def compose_name(path, time, index, data):
    return path+'_'.join([time+'%02d' % index, data.label+'.pkl.gz'])


def file_name(path, data):
    time = strftime('%Y%m%d%H%M%S')
    index = 0
    while os.path.isfile(compose_name(path, time, index, data)):
        index += 1
    return compose_name(path, time, index, data)
    

def save_data(path, data):
    if not os.path.exists(path):
        os.makedirs(path)
    with gz.open(file_name(path, data), 'wb') as f:
        dump(data, f)


def rmse(tgt, opt):
    return sqrt(mean(dot(subtract(tgt, opt), subtract(tgt, opt))))


class Data:
    def __init__(self, label, params, stimulus, conn_rmses, data, rmses, weights, dims):

        self.label = label
        self.params = params
        self.stimulus = stimulus
        self.conn_rmses = conn_rmses
        self.data = data
        self.dims = dims
