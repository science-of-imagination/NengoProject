import nengo
from utils.collect import Data
from data import load_img, load_mini_mnist
from utils.encoders import normalized_random_gabor_encoders
from numpy import array, ones, sqrt, amax, amin, subtract, divide
from numpy.linalg import norm
import os


def run(N, img_path, w, h):

    N, w, h = int(N), int(w), int(h)
    dims = (w, h)

    print 'Loading image.'
    img = load_img(img_path, dims)
                              
    print 'Initializing encoders.'
    encs = normalized_random_gabor_encoders(w, N)

    print 'Building model.'
    with nengo.Network() as net:

        def stim_func(t):
            if t<0.1:
                return img
            else:
                return [0]*len(img)

        ipt = nengo.Node(stim_func)
        ens = nengo.Ensemble(N,
                             dimensions=len(img),
                             encoders=encs,
                             radius=sqrt(dims[0]*dims[1]))

        nengo.Connection(ipt, ens, synapse=1, transform=1)
        nengo.Connection(ens, ens, synapse=1)

        probe = nengo.Probe(ens, attr='decoded_output',
                            synapse=0.1)
        
    print 'Running simulation.'
    sim = nengo.Simulator(net)
    sim.run(0.2)

    print 'Collecting data.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, img_path, w, h),
                img,
                array([opt for opt in sim.data[probe]]),
                dims)
