'''Utility to collect data on simulations'''

import gzip as gz
from cPickle import dump
from time import strftime


def save_data(path, data):
    filename = path+strftime('%Y%m%d%H%M%S')+data.label+'.pkl.gz'
    with gz.open(filename, 'wb') as f:
        dump(data, f)


class Data:

    def __init__(self, label, stimulus, data):

        self.label = label
        self.stimulus = stimulus
        self.data = data


class Model:

    def __init__(self, model_consturctor):

        self.model, self.inputs, self.probes = model_constructor()

    def run(duration):
        pass

        
