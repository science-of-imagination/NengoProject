'''Utility to collect data on simulations'''

import gzip as gz
import os
from cPickle import dump
from time import strftime
from nengo import Simulator
from visualize import figure_from_vector
from matplotlib.pyplot import show, savefig, clf


def save_data(path, data):
    filename = strftime('%Y%m%d%H%M%S')+data.label
    with gz.open(''.join([path, filename, '.pkl.gz']), 'wb') as f:
        dump(data, f)
    return filename


class Data:

    def __init__(self, label, stimulus, data):

        self.label = label
        self.stimulus = stimulus
        self.data = data


class Model:

    def __init__(self, model_constructor, params, ipts):
        #Only accommodates a single probe at present
        #Not sure how Data call would have to be to deal with multiple probes
        self.params = params
        self.name = ''.join(['mnist0_', str(params['ens_size'])])
        self.inputs = ipts
        self.model, self.probe = model_constructor(**params)
        self.path = './model_outputs/feed_forward_gabor_canvas/'

    def run(self, duration, printimgs=True, avgimg=True):
        sim = Simulator(self.model)
        sim.run(duration)
        data = Data(self.name, self.inputs, sim.data[self.probe])
        #Note, this will not work if not called from dev folder
        filename = save_data(self.path, data)
        print 'Model ran successfully'
        if printimgs:
            print 'Generating images...'
            directory = ''.join([self.path, 'imgs/', filename])
            if not os.path.exists(directory):
                os.makedirs(directory)
            for i in range(len(data.data)):
                figure_from_vector(data.data[i], self.params['canvas_size'])
                savefig(self.path+'/imgs/'+filename+'/'+str(i)+'.png')
                if i%100==0:
                    print 'Saved img '+str(i)
                clf()
            print 'Images generated'
        if avgimg:
            print 'Generating average...'
            ##
            #Captures only the input part of the set (as opposed to stabilized part)
            ##
            avg = sum(data.data[:100, :])/len(data.data)
            figure_from_vector(avg, self.params['canvas_size'])
            show()
        print 'Done'
        

        
