import nengo
from utils.collect import Data, rmse
from data import load_img, load_mini_mnist
from utils.encoders import mk_bgbrs, normalized_random_gabor_encoders, mk_gbr_eval_pts
from numpy import array, ones, sqrt, subtract, dot, where, zeros, flipud
from numpy.linalg import norm, svd
import os
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix

def run(img,pee,comp_encs,comp_evl,basis,D,N, n_eval_pts,w, h, t=0.2):
    
    print "ASD", N, n_eval_pts,w,h,t
    N, n_eval_pts, w, h, t = int(N), int(n_eval_pts), int(w), int(h), float(t)
    dims = (w, h)

    print 'Loading image.'
    img = img.flatten()
    img = img/norm(img)
                              

    
    def compress(original):
        return dot(original, basis)

    def uncompress(compressed):
        return dot(basis, compressed.T).T
    
    print 'Compressing img.'
    comp_img = compress(img)



    print 'Building model.'
    with nengo.Network() as net:

        def stim_func(t):
            if t<0.1:
                return comp_img
            else:
                return zeros(D)

        neuron_type = nengo.LIFRate() 

        ipt = nengo.Node(stim_func)
        ens = nengo.Ensemble(N,
                             dimensions=D,
                             encoders=comp_encs,
                             eval_points=comp_evl,
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
    #print 'Recording connection weights.'
    #weights = dot(encs, sim.data[conn].decoders)
    print 'Recording rmses per sample.'
    rmses = array([rmse(img, uncompress(j)) for j in sim.data[probe]])

    print 'Simulation finished.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, n_eval_pts, w, h),
                img,
                conn_rmse,
                uncompress(array([opt for opt in sim.data[probe]])),
                rmses,
                None,#weights,
                dims,
                pee)
