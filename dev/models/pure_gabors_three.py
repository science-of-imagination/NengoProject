import nengo
from data import load_img
from utils.collect import Data, rmse
from utils.encoders import mk_bgbrs
from numpy import array, dot, convolve
from numpy.linalg import norm, pinv
import os

import time
start = time.time()


def now():
    return '%7.3f' % (time.time() - start)

def run(img, pee,encs,decs, N, n_eval_pts,  w, h):

    N, w, h = int(N), int(w), int(h)
    dims = (w, h)
    img = img.flatten()
    img = img/norm(img)
    #print 'Loading image.'
    #img = load_img(img_path, dims)
                              

    #print decs.shape
    print str(now()) + ' Encoding image.'
    coeffs = dot(encs, img)
    print coeffs.shape
    print str(now()) + ' Decoding image.'
    tsfmd = dot(decs, coeffs)    
    print str(now()) + ' Recording rmses per sample.'
    rmses = rmse(img, tsfmd)
    print rmses

    print str(now()) + ' Simulation finished.'
    return Data(os.path.basename(__file__).strip('.py').strip('.pyc'),
                (N, n_eval_pts , w, h),
                img,
                None,
                [tsfmd],
                rmses,
                None,
                dims,
                pee)

