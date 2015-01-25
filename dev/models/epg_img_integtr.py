import nengo
from utils.collect import Data
from data import load_img, load_mini_mnist
from utils.encoders import normalized_random_gabor_encoders
from numpy import array, ones, sqrt, amax, amin, subtract, divide
from numpy.linalg import norm
import os


def run(Ne, npe, img_path, w, h):

    Ne, npe, w, h = int(Ne), int(npe), int(w), int(h)
    dims = (w, h)

    print 'Loading image.'
    img = load_img(img_path, dims)
                              
    print 'Initializing encoders.'
    encs = normalized_random_gabor_encoders(w, Ne)

    print 'Building model.'
    with nengo.Network() as net:

        def stim_func(t):
            if t<0.1:
                return img
            else:
                return [0]*len(img)

        ipt = nengo.Node(stim_func)
        opt = nengo.Node(size_in=len(img))
        for i in range(Ne):
            ens = nengo.Ensemble(npe,
                                 dimensions=len(img),
                                 encoders=[encs[i]]*npe,
                                 radius=sqrt(dims[0]*dims[1]))

            nengo.Connection(ipt, ens, synapse=1, transform=1)
            nengo.Connection(ens, ens, synapse=1)
            nengo.Connection(ens, opt)

        probe = nengo.Probe(opt, attr='output',
                            synapse=0.1)
    print opt.probeable
        
    print 'Running simulation.'
    sim = nengo.Simulator(net)
    sim.run(0.2)

    print 'Collecting data.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (Ne, npe, img_path, w, h),
                img,
                array([op for op in sim.data[probe]]),
                dims)
