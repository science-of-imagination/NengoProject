import nengo
from data import load_img
from utils.collect import Data, rmse
from utils.encoders import mk_bgbrs
from numpy import array, dot
from numpy.linalg import norm
import os


def run(N, img_path, w, h):

    N, w, h = int(N), int(w), int(h)
    dims = (w, h)

    print 'Loading image.'
    img = load_img(img_path, dims)
                              
    print 'Initializing encoders.'
    encs = array([j.flatten()/norm(j) for j in
                  mk_bgbrs(N/2, dims, dims[0]*4)])

    tsfmd = dot(encs.T, dot(encs, img)) 
    tsfmd = tsfmd/norm(tsfmd)    
    print 'Recording rmses per sample.'
    rmses = rmse(img, tsfmd)
    print rmses

    print 'Simulation finished.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, img_path, w, h),
                img,
                None,
                None,
                rmses,
                None,
                dims)
