import nengo
from utils.collect import Data, rmse
from data import load_img, load_mini_mnist
from utils.encoders import mk_bgbrs, normalized_random_gabor_encoders, mk_gbr_eval_pts
from numpy import array, ones, sqrt, amax, amin, subtract, divide, dot
from numpy.linalg import norm
import os


def run(N, n_eval_pts, img_path, w, h, t=0.05):

    N, n_eval_pts, w, h, t = int(N), int(n_eval_pts), int(w), int(h), float(t)
    dims = (w, h)

    print 'Loading image.'
    img = load_img(img_path, dims)
                              
    print 'Initializing encoders.'
    encs = array(mk_bgbrs(N/2, dims, dims[0]/float(8)))

    print 'Initializing eval points.'
    eval_points = mk_gbr_eval_pts(n_eval_pts, dims[0])
    
    print 'Building model.'
    with nengo.Network() as net:

        def stim_func(t):
            if t<0.1:
                return img
            else:
                return [0]*len(img)

        neuron_type = nengo.LIFRate() 

        ipt = nengo.Node(stim_func)
        ens = nengo.Ensemble(N,
                             dimensions=len(img),
                             encoders=encs,
                             eval_points=eval_points,
                             neuron_type=neuron_type)

        nengo.Connection(ipt, ens, synapse=None, transform=1)
        probe = nengo.Probe(ens, attr='decoded_output',
                            synapse=0.01)
        
    print 'Running simulation.'
    sim = nengo.Simulator(net)
    sim.run(t)
    print 'Recording rmses per sample.'
    rmses = array([rmse(img, j) for j in sim.data[probe]])
    print 'Error on last frame :'+str(rmses[49])

    print 'Simulation finished.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, eval_points, img_path, w, h),
                img,
                None,
                array([opt for opt in sim.data[probe]]),
                rmses,
                None,
                dims)