import nengo
from utils.collect import Data
from data import load_img, load_mini_mnist
from utils.encoders_search import normalized_random_gabor_encoders
from numpy import array, zeros, tile, sqrt, sum
from numpy.linalg import norm
import os
from multiprocessing import Pool


def run(N, img_path, w, h):

    N = int(N)
    w = int(w)
    h = int(h)
    dims = (w, h)
    
    img = load_img(img_path, dims)
    
    encs = cans_encoders
    
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

        probe = nengo.Probe(ens, attr='decoded_output', synapse=0.1)

    sim = nengo.Simulator(net)
    sim.run(0.2)
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, img_path), img,
                array([opt for opt in sim.data[probe]]),
                dims)

