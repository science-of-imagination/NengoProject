'''
Model goal:
    Load an image to an ensemble and recover it.

Network architecture:

 ipt -> ens -> opt
           | 
         probe
'''


from data import load_mini_mnist
from numpy import identity
import nengo

def modelbuilder(canvas_size, ens_size, sigmarng, encoderfunc, iptfunc):
    encs = encoderfunc(canvas_size, ens_size, sigmarng)

    with nengo.Network(label="Net_1") as Net_1:
        ipt = nengo.Node(iptfunc)
        ens = nengo.Ensemble(ens_size,
                             dimensions=canvas_size**2,
                             encoders=encs,
                             label="ens")
        out = nengo.Node(size_in=canvas_size**2)
        
        trans=identity(2**2)#*0.5

        nengo.Connection(ipt, ens, synapse=0.4) #transform=trans)
        nengo.Connection(ens, out, synapse=0.4) #transform=trans)
        nengo.Connection(ens, ens, synapse=0.4)    
        
        probe = nengo.Probe(ens, attr="decoded_output")

    return Net_1, probe
