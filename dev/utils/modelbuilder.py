'''
Model goal:
    Load an image to an ensemble and recover it.

Network architecture:

 ipt -> ens -> opt
           | 
         probe
'''

from encoders import normalized_random_gabor_encoders
from data import load_mini_mnist
import nengo

def modelbuilder(canvas_size, ens_size, sigmarng, iptfunc):
    encs = normalized_random_gabor_encoders(canvas_size, ens_size, sigmarng)

    with nengo.Network(label="Net_1") as Net_1:
        ipt = nengo.Node(iptfunc)
        ens = nengo.Ensemble(ens_size,
                             dimensions=canvas_size**2,
                             encoders=encs,
                             label="ens")
        out = nengo.Node(size_in=canvas_size**2)
        
        

        nengo.Connection(ipt, ens)
        nengo.Connection(ens, out)
        nengo.Connection(ens, ens)    
        
        probe = nengo.Probe(ens, attr="decoded_output")

    return Net_1, probe
