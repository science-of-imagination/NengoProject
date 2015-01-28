import nengo
from utils.collect import Data
from data import load_img, load_mini_mnist
from utils.encoders_search import normalized_random_gabor_encoders
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
        theta='uniform(0,2*pi)'
        lambd='0.001'
        sigma = 0 + (1/100.0)   
        #lambd_end = 0 + (i/1000.0)
        #lambd = 'uniform(2/canvas_size,' + repr(lambd_end) + ')'
        encs = normalized_random_gabor_encoders(dims[0], 1,thetaSTR=theta,lambdSTR=lambd,sigmaSTR=repr(sigma),x_offSTR='uniform(-1,1)',
                                                y_offSTR='uniform(-1,1)')
        #encs = normalized_random_gabor_encoders(dims[0], 1)

        with nengo.Network() as net:

            def stim_func(t):
                if t<0.1:
                    return img
                else:
                    return [0]*len(img)

            ipt = nengo.Node(stim_func)
            ens = nengo.Ensemble(N,
                                 dimensions=len(img),
                                 encoders=tile(encs, (N,1)),
                                 radius=sqrt(dims[0]*dims[1]))

            nengo.Connection(ipt, ens, synapse=1, transform=1)
            nengo.Connection(ens, ens, synapse=1)

            probe = nengo.Probe(ens, attr='decoded_output',
                                synapse=0.1)
            
        sim = nengo.Simulator(net)
        sim.run(0.2)
    except RuntimeError as e:
        return sub_run((N, img, dims))

    return array(sim.data[probe])


def run(Ne, npe, img_path, w, h):

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
                     (Ne, npe, img_path, w, h),
                     img,
                     array(opt),
                     dims)
