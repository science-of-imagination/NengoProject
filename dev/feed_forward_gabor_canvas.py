'''
Model goal:
    Load an image to an ensemble and recover it.

Network architecture:

 ipt -> ens -> opt
           | 
         probe
'''

from utils.encoders import normalized_random_gabor_encoders
from utils.collect import save_data, Data
from data import load_mini_mnist
import nengo

canvas_size = 28
ens_size = 2000
stimulus = 0

#build encoders
encs = normalized_random_gabor_encoders(canvas_size, ens_size)

#get inputs
ipts = load_mini_mnist('train')[:5]

with nengo.Network(label="Net_1") as Net_1:
    ipt = nengo.Node(list(ipts[stimulus]))
    ens = nengo.Ensemble(ens_size,
                         dimensions=canvas_size**2,
                         encoders=encs,
                         label="ens")
    out = nengo.Node(size_in=canvas_size**2)
    
    nengo.Connection(ipt, ens)
    nengo.Connection(ens, out)
    
    probe = nengo.Probe(ens, attr="decoded_output")
    
sim = nengo.Simulator(Net_1)
sim.run(0.5)
data = Data('mmnist0', ipts[stimulus], sim.data[probe])
save_data('./model_outputs/feed_forward_gabor_canvas/', data)
print 'Model ran successfully'
