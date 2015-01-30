import nengo
from utils.collect import Data, rmse
from data import load_img, load_mini_mnist
from utils.encoders import mk_bgbrs, normalized_random_gabor_encoders
from numpy import array, ones, sqrt, amax, amin, subtract, divide, dot
from numpy.linalg import norm, svd
import os


def run(N, n_eval_pts, img_path, w, h):

    N, n_eval_pts, w, h = int(N), int(n_eval_pts), int(w), int(h)
    dims = (w, h)

    print 'Loading image.'
    img = load_img(img_path, dims)
                              
    print 'Initializing encoders.'
    encs = array(normalized_random_gabor_encoders(w, N))
    #encs = array([j.flatten()/norm(j) for j in
    #              mk_bgbrs(N/2, dims, dims[0]*4)])

    print 'Initializing eval points.'
    eval_points = normalized_random_gabor_encoders(w, n_eval_pts)
    #eval_points = array([j.flatten()/norm(j) for j in
    #              mk_bgbrs(n_eval_pts/2, dims, dims[0]*4)])

    
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
        conn = nengo.Connection(ens, ens, synapse=1)

        probe = nengo.Probe(ens, attr='decoded_output',
                            synapse=0.01)
        
    print 'Running simulation.'
    sim = nengo.Simulator(net)
    sim.run(0.2)

    print 'Collecting connection error.'
    conn_rmse = norm(sim.data[conn].solver_info['rmses'])
    print 'Collecting connection weights.'
    weights = dot(encs, sim.data[conn].decoders)
    print 'Collecting rmses per sample.'
    rmses = array([rmse(img, j) for j in sim.data[probe]])

    print 'Collecting data.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, eval_points, img_path, w, h),
                img,
                conn_rmse,
                array([opt for opt in sim.data[probe]]),
                rmses,
                weights,
                dims)