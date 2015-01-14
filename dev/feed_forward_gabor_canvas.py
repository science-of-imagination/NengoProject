'''
Model goal:
    Load an image to an ensemble and recover it.

Network architecture:

 ipt -> ens -> opt
           | 
         probe
'''

from utils.encoders import *
from utils.collect import save_data, Data
from data import load_mini_mnist
import nengo

canvas_size = 28
ens_size = 4105
stimulus = 0

#build encoders
encs = normalized_random_patch_encoders(canvas_size, ens_size)

#get inputs
ipts = load_mini_mnist('train')[:5]

with nengo.Network(label="Net_1") as Net_1:

    def loop_func(v):
        return 0.01*v

    def stim_func(t):
        if t<0.1:
            return list(ipts[stimulus])
        else:
            return [0]*784

    ipt = nengo.Node(stim_func)
    ens = nengo.Ensemble(ens_size,
                         dimensions=canvas_size**2,
                         encoders=encs,
                         label="ens")
    out = nengo.Node(size_in=canvas_size**2)
    #nengo.Connection(ipt, ens)
    nengo.Connection(ipt, ens, synapse=0.1)
    nengo.Connection(ens, out)
    nengo.Connection(ens, ens, synapse=0.1, function=loop_func)   
   # nengo.Connection(ens,ens) 
    
    probe = nengo.Probe(ens, attr="decoded_output")

    
    
sim = nengo.Simulator(Net_1)

sim.run(.50)


data = Data('patches_1_' + repr(ens_size), ipts[stimulus], sim.data[probe])
save_data('./model_outputs/feed_forward_gabor_canvas/', data)
print 'Model ran successfully'
