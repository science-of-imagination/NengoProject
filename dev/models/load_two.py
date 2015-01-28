import nengo
from utils.collect import Data, rmse
from data import load_img, load_mini_mnist
from utils.encoders import mk_bgbrs, normalized_random_gabor_encoders, mk_gbr_eval_pts
from numpy import array, ones, sqrt, amax, amin, subtract, divide, dot
from numpy.linalg import norm
import os


def run(N, n_eval_pts, t=0.2):

    N, n_eval_pts, t = int(N), int(n_eval_pts), float(t)
    dims = (28, 28)

    print 'Loading image.'
    img_1 = load_img('./data/4_.png', dims)
    img_2 = load_img('./data/_2.png', dims)
    
    print 'Initializing encoders.'
    encs = array([j.flatten()/norm(j) for j in
                  mk_bgbrs(N/2, dims, dims[0]*4)])

    print 'Initializing eval points.'
    eval_points = mk_gbr_eval_pts(n_eval_pts, dims[0])
    
    print 'Building model.'
    with nengo.Network() as net:

        def stim_func(t):
            if t<0.05:
                return img_1
            elif t<0.1:
                return img_2
            else:
                return [0]*784

        neuron_type = nengo.LIFRate() 

        ipt = nengo.Node(stim_func)
        ens = nengo.Ensemble(N,
                             dimensions=784,
                             encoders=encs,
                             eval_points=eval_points,
                             neuron_type=neuron_type)

        nengo.Connection(ipt, ens, synapse=None, transform=1)
        conn = nengo.Connection(ens, ens, synapse=1)

        probe = nengo.Probe(ens, attr='decoded_output',
                            synapse=0.01)
        
    print 'Running simulation.'
    sim = nengo.Simulator(net)
    sim.run(t)
    print 'Connection RMSE: '+str(norm(sim.data[conn].solver_info['rmses']))
    print 'Recording connection error.'
    conn_rmse = norm(sim.data[conn].solver_info['rmses'])
    print 'Recording connection weights.'
    weights = dot(encs, sim.data[conn].decoders)

    print 'Simulation finished.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, eval_points),
                (img_1,img_2),
                conn_rmse,
                array([opt for opt in sim.data[probe]]),
                None,
                weights,
                dims)
