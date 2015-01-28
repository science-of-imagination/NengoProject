import nengo
from utils.collect import Data, rmse
from data import load_img, load_mini_mnist
from utils.encoders import mk_bgbrs, normalized_random_gabor_encoders, mk_gbr_eval_pts
from numpy import array, ones, sqrt, subtract, dot, where, zeros
from numpy.linalg import norm, svd
import os
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix

def run(N, n_eval_pts, img_path, w, h, t=0.2):

    N, n_eval_pts, w, h, t = int(N), int(n_eval_pts), int(w), int(h), float(t)
    dims = (w, h)

    print 'Loading image.'
    img = load_img(img_path, dims)
                              
    print 'Initializing encoders.'
    encs = array([j.flatten()/norm(j.flatten()) for j in
                  mk_bgbrs(N/2, dims, dims[0]*4)])

    print 'Initializing eval points.'
    eval_points = mk_gbr_eval_pts(n_eval_pts, dims[0])

    print 'Initializing SVD compression.'
    U, S, V = svds(encs.T, N-1)
    print S.shape
    import pylab
    pylab.plot(S)
    pylab.show()
    #print where(S<S[0]*0.01)
    D = where(S<S[0]*0.01)[0][0]
    print D
    basis = U[:,:D]
    
    def compress(original):
        print original.shape
        print basis.shape
        return dot(original, basis)

    def uncompress(compressed):
        return dot(basis, compressed.T).T
    
    print 'Building model.'
    with nengo.Network() as net:

        def stim_func(t):
            if t<0.1:
                return compress(img)
            else:
                return zeros(D)

        neuron_type = nengo.LIFRate() 

        ipt = nengo.Node(stim_func)
        ens = nengo.Ensemble(N,
                             dimensions=D,
                             encoders=compress(encs),
                             eval_points=compress(eval_points),
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
    print 'Recording rmses per sample.'
    rmses = array([rmse(img, j) for j in sim.data[probe]])

    print 'Simulation finished.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, eval_points, img_path, w, h),
                img,
                conn_rmse,
                uncompress(array([opt for opt in sim.data[probe]])),
                rmses,
                weights,
                dims)
