import cPickle 
import gzip
import matplotlib.pyplot as plt
import numpy
import scipy.sparse
from sparsesvd import sparsesvd
try:
    from PIL import Image
except ImportError:
    import Image

img_size = 28
n_B = 2000
n_bases = 80

# Use MNIST dataset, many images.
# Load the dataset
f = gzip.open('Pictures/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# Get Imgs
Imgs = numpy.array([img/numpy.linalg.norm(img) for img in train_set[0]]).T
Imgs_csc = scipy.sparse.csc_matrix(Imgs)

ut, S, vt = sparsesvd(Imgs_csc, n_bases)
M = numpy.diag([numpy.linalg.norm(ut[i,:]) for i in range(ut.shape[0])])
#W is M inverse
UW = numpy.dot(ut.T, numpy.linalg.inv(M))
MSvt = numpy.dot(M, numpy.dot(numpy.diag(S), vt))

import nengo

def render(v):
    return numpy.dot(UW, v)

Net_1 = nengo.Network(label="Net_1")
with Net_1:
    ipt = nengo.Node(list(MSvt[:,4]))
    
    renderer = nengo.Ensemble(img_size**2, 
                              dimensions=n_bases, 
                              encoders=UW, 
                              label="renderer")#, neuron_type=nengo.Direct())
    ibuffer = nengo.Ensemble(n_B, 
                             dimensions=img_size**2, 
                             encoders=numpy.array([encoder.flatten() for encoder in encoders_B]), 
                             label="ibuffer",)#, neuron_type=nengo.Direct())
    out = nengo.Node(size_in=img_size**2)
    
    nengo.Connection(ipt, renderer)
    nengo.Connection(renderer, ibuffer, function=render, synapse=0.01)
    nengo.Connection(ibuffer, ibuffer, function=lambda x: 0.01*x, synapse=0.01)
    
    probeIbuffer_input = nengo.Probe(ibuffer, attr="input")
    probeIbuffer_output = nengo.Probe(ibuffer, attr="decoded_output")
    
sim = nengo.Simulator(Net_1)
sim.run(0.5)
sim.run(0.5)
ibuffer_inputData = sim.data[probeIbuffer_input]
ibuffer_outputData = sim.data[probeIbuffer_output]    
