import nengo
from utils.collect import Data
from data import load_img, load_mini_mnist
from utils.encoders import pixel_encoders
from numpy import array, zeros, tile, sqrt, cos, sin, log, pi, exp, linspace, meshgrid
from numpy.linalg import norm
from random import uniform, choice
import os


def mk_mesh(dims, x_offset, y_offset):
    '''Create a mesh to represent the space from which information is
    available to an encoder.'''

    x = linspace(-1, 1, dims[0])
    y = linspace(-1, 1, dims[1])
    X, Y = meshgrid(x, y)
    X = X - x_offset
    return X - x_offset, Y + y_offset


def rotate_mesh(theta, mesh):
    '''Rotate the mesh by theta, for asymmetric encoders--where orientation
    matters.'''
    return (mesh[0]*cos(theta)+mesh[1]*sin(theta),
            -mesh[0]*sin(theta)+mesh[1]*cos(theta))

def bio_gbr(dims, x_off, y_off, theta, f, phi, psi=0):
    '''Returns a quadrature pair of biologically inspired gabor placed on
    a pixel canvas of dimensions dims. (Short axis of envelope is aligned
    with the axis along which the waves travel;
    |long axis of envelope| = 2*|short axis of envelope|.)

    Note: Square canvas suggested.

    x_offset: position of the center of gabor along the x-axis of the image

    y_offset: position of the center of gabor along the y-axis of the image

    WARNING: x_offset, y_offset must be in [-1,1]

    theta: orientation of the gabor

    f: main frequency of gabor in Hz/canvas_length

    phi: Half-amplitude spatial frequency bandwidth of gabor.

    psi: phase of the gabor.'''

    #Compute kappa, sigma
    kappa = sqrt(2*log(2))*(2.0**phi+1)/(2.0**phi-1)
    sigma = kappa/(2*pi*f)
    f = 2*f
    
    #Change of basis, centered on center of gbr, rotated to align with the
    # direction of wave propagation
    X, Y = rotate_mesh(theta, mk_mesh(dims, x_off, y_off))

    #Compute parts of the gabor
    envelope = (sqrt(2*pi)/sigma)*exp(-(pi/(2*sigma))*(4*(X**2) + Y**2))
    re = cos(2*pi*f*X+psi)-exp(-kappa**2/2.0)
    im = sin(2*pi*f*X+psi)

    return envelope*re, envelope*im


#Lee (1996) says that neuro evidence points at 1.5 octaves for phi, and 2-3
# voices per octave. That the system looks at 3-5 octaves. That observations
# yield around 20 different directions represented.
# All this is implemented below.

def mk_bgbrs(n_pairs,
             dims,
             F_max,
             octaves=5,
             N=3,
             x_off=lambda:uniform(-1,1),
             y_off=lambda:uniform(-1,1),
             theta=lambda:choice([2*pi*i/20 for i in range(20)]),
             phi=lambda:1.5,
             psi=lambda:uniform(0,2*pi)):

    n_steps = octaves*N
    Fs = []
    for i in range(octaves*N):
        Fs.append((2**(-i/float(N)))*F_max)

    f_ch = lambda:choice(Fs)

    gbrs = []
    for i in range(n_pairs):
        nxt = bio_gbr(dims, x_off(),  y_off(), theta(), f_ch(), phi(), psi())
        gbrs.append(nxt[0])
        gbrs.append(nxt[1])
    return gbrs


def sub_run(N, img, dims, i):


    encs = mk_bgbrs(1, dims, dims[0]/float(16))

    out = zeros((200, len(img)))
    for e in encs:
        with nengo.Network() as net:

            def stim_func(t):
                if t<0.1:
                    return img
                else:
                    return [0]*len(img)

            ipt = nengo.Node(stim_func)
            ens = nengo.Ensemble(N,
                                 dimensions=len(img),
                                 encoders=tile(e.flatten(), (N,1)),
                                 radius=sqrt(dims[0]*dims[1]))

            nengo.Connection(ipt, ens, synapse=1, transform=1)
            nengo.Connection(ens, ens, synapse=1)

            probe = nengo.Probe(ens, attr='decoded_output',
                                synapse=0.1)
            
        sim = nengo.Simulator(net)
        sim.run(0.2)
        out += array(sim.data[probe])

    return out


def run(npe, img_path, w, h, N):

    npe, w, h, N = int(npe), int(w), int(h), int(N)
    dims = (w, h)
    
    print 'Loading image.'
    img = load_img(img_path, dims)

    print 'Running sub-simulations.'
    opt = zeros((200,len(img)))
    itetr = range(N)
    errors = 0
    pixes = 0
    for n in itetr:
        print 'Sub-simulation %d of %d' % (n+1,N)
        try:
            opt += sub_run(npe, img, dims, pixes)
            pixes += 1
        except RuntimeError as e:
            itetr.append(itetr[-1]+1)
            errors += 1
    print 'There were %d errors.' % errors

    print 'Collecting data.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                     (npe, img_path, w, h),
                     img,
                     array(opt),
                     dims)
