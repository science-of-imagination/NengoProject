import nengo
from utils.collect import Data, format_output
from data import load_img
from utils.encoders import normalized_random_gabor_encoders
from numpy import array
import os

def run(N, img_path, w, h):

    N = int(N)
    w = int(w)
    h = int(h)
    dims = (w, h)
    
    img = load_img(img_path, dims)

    
    encs = normalized_random_gabor_encoders(w, N)
    
    def stim_func(t):
        if t<0.1:
            return img
        else:
            return [0]*len(img)

    with nengo.Network() as net:
        ipt = nengo.Node(stim_func)
        ens = nengo.Ensemble(N,
                            dimensions=len(img))

        nengo.Connection(ipt, ens, synapse=1, transform=1)
        nengo.Connection(ens, ens, synapse=1)

        probe = nengo.Probe(ens, attr='decoded_output', synapse=0.01)

    sim = nengo.Simulator(net)
    sim.run(0.2)
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, img_path), img,
                array([format_output(opt) for opt in sim.data[probe]]),
                dims)
