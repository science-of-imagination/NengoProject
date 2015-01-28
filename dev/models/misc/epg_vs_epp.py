import nengo
from utils.collect import Data
from data import load_img, load_mini_mnist
from utils.encoders import normalized_random_gabor_encoders, pixel_encoders
from numpy import array, zeros, tile, sqrt
from numpy.linalg import norm
import os


#Idea: epg_img_integtr sums all the outputs in the last stage
# but building the whole thing on its own is to memory costly.
# so do each individual loop in a separate simulation. it doesnt
# make a difference since the separate ensembles dont talk to
# eachother anyway.


def sub_run(pVg, N, img, dims, i):

    if pVg == 'gbr':
        encs = normalized_random_gabor_encoders(dims[0], 1)
    elif pVg == 'pix':
        encs = pixel_encoders(dims[0])[i]

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

    return array(sim.data[probe])


def run(pVg, npe, img_path, w, h):

    npe, w, h = int(npe), int(w), int(h)
    dims = (w, h)
    
    print 'Loading image.'
    img = load_img(img_path, dims)

    print 'Running sub-simulations.'
    opt = zeros((200,len(img)))
    itetr = range(dims[0]*dims[1])
    errors = 0
    pixes = 0
    for n in itetr:
        print 'Sub-simulation %d of %d' % (n+1,dims[0]*dims[1])
        try:
            opt += sub_run(pVg, npe, img, dims, pixes)
            pixes += 1
        except RuntimeError as e:
            itetr.append(itetr[-1]+1)
            errors += 1
    print 'There were %d errors.' % errors

    print 'Collecting data.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                     (pVg, npe, img_path, w, h),
                     img,
                     array(opt),
                     dims)