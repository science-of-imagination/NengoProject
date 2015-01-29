import nengo
from utils.collect import Data
from data import load_img, load_mini_mnist
from utils.encoders_search import *
from numpy import array, zeros, tile, sqrt, sum
from numpy.linalg import norm
import os
from multiprocessing import Pool

#Idea: epg_img_integtr sums all the outputs in the last stage
# but building the whole thing on its own is to memory costly.
# so do each individual loop in a separate simulation. it doesnt
# make a difference since the separate ensembles dont talk to
# eachother anyway.


def sub_run((N, img, dims)):
    
    try:
        def compress(original):
            print original.shape
            print basis.shape
            return dot(original, basis)

        def uncompress(compressed):
            return dot(basis, compressed.T).T


        encs = array([j.flatten()/norm(j.flatten()) for j in
                  mk_bgbrs(N/2, dims, dims[0]*4)])
        eval_points = mk_gbr_eval_pts(n_eval_pts, dims[0])

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
            
        sim = nengo.Simulator(net)
        sim.run(0.2)
    except RuntimeError as e:
        return sub_run((N, img, dims))

    return array(sim.data[probe])


def run(N, n_eval_pts, img_path, w, h, t=0.2):

    Ne, npe, w, h = int(Ne), int(npe), int(w), int(h)
    dims = (w, h)
    
    print 'Loading image.'
    img = load_img(img_path, dims)

    print 'Running sub-simulations.'
    opt = zeros((200,len(img)))
    itetr = range(Ne)
    errors = 0
    pool = Pool(processes=4)

    for p in range(Ne/4):
        result = pool.map(sub_run, [(npe, img, dims)]*4)
        for r in result:
            opt += r
     

    result = pool.map(sub_run, [(npe, img, dims)]*(Ne%4))
    for r in result:
        opt += r

    print 'Collecting data.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, eval_points, img_path, w, h),
                img,
                conn_rmse,
                uncompress(array([opt for opt in sim.data[probe]])),
                rmses,
                weights,
                dims)
