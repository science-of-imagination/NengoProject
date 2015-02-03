import nengo
from data import load_img
from utils.collect import Data, rmse
from utils.encoders import mk_bgbrs
from numpy import array, dot, convolve
from numpy.linalg import norm, pinv
import os



def run(img,pee,N, n_eval_pts,  w, h):

    N, w, h = int(N), int(w), int(h)
    dims = (w, h)
    img = img.flatten()
    img = img/norm(img)
    #print 'Loading image.'
    #img = load_img(img_path, dims)
                              
    print 'Initializing encoders.'
    encs = array(mk_bgbrs(N/2, dims, 4))
    decs = pinv(encs)
    print decs.shape
    coeffs = dot(encs, img)
    print coeffs.shape
    tsfmd = dot(decs, coeffs)    
    print 'Recording rmses per sample.'
    rmses = rmse(img, tsfmd)
    print rmses

    print 'Simulation finished.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, n_eval_pts , w, h),
                img,
                None,
                [tsfmd],
                rmses,
                None,
                dims,
                pee)

