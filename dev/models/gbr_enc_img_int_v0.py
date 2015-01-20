import nengo
from utils.collect import Data
from data import load_img
from utils.encoders import normalized_random_gabor_encoders

def run(N, img_path, w, h):

    N = int(N)
    w = int(w)
    h = int(h)
    dims = (w, h)
    
    img = load_img(img_path, dims)

    encs = 

    def stim_func(t):
        if t<0.1:
            return img
        else:
            return [0]*len(img)

    with nengo.Network() as net:
        ipt = nengo.Node(stim_func)
        ens = nengo.Ensemble(N,
                            dimensions=len(img))

        nengo.Connection(ipt, ens, synapse=1, transform=1)
        nengo.Connection(ens, ens, synapse=1)

        probe = nengo.Probe(ens, attr='decoded_output', synapse=0.01)

    sim = nengo.Simulator(net)
    sim.run(0.2)
    return Data('dflt_enc_img_int', (N, img_path), img, sim.data[probe], dims)
