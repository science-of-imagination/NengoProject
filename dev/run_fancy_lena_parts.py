'''This module is used to run nengo models. See README file.'''

#out_path determines where model data will be saved. You must create
# cfg.py and set out_path to be the path to the directory you want
# your models to be stored in. You do not need to manually create this
# directory.
from cfg import out_path
from utils.collect import save_data
import sys
import getopt
from data import load_data
from numpy import array, ones, sqrt, subtract, dot, where, zeros, flipud
from utils.encoders import mk_bgbrs, normalized_random_gabor_encoders, mk_gbr_eval_pts
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
import scipy
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt


def load_params(name, opt=None):
    with open(name, 'rb') as pm:
        lines = pm.readlines()
        if not opt:
            lines = [lines[-1]]
    return [line.split() for line in lines]


def param_opt(opts):
    if ('-l', '') in opts:
        return True


def param_file(args):
    if len(args)<2:
        return './models/'+args[0]+'.params'
    else:
        return './models/'+args[1]+'.params'

def patchify(img, patch_shape):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
        
    #shape = (3,3,x,y)
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    print "strides",strides
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

def run_model(model_name, params):
    exec 'from models.%s import run' % model_name
    opts = []
    runs = len(params)
    lena = scipy.misc.lena()
    lena = lena.repeat(2,axis=0).repeat(2,axis=1)

#    lena.resize((1024,1024))

    #plt.imshow(lena,cmap='gray')
    #plt.show()
    #return 0

    lena_parts = patchify(lena,(32,32))
    lena_parts = np.ascontiguousarray(lena_parts)
    parts = []
    for i in range(len(lena_parts)):
        for j in range(len(lena_parts[i])):
            if not(i%32) and not(j%32):
                parts.append(lena_parts[i][j])
    lena_parts = np.array(parts)
    #print lena_parts[0].shape
    #print lena_parts.shape
    #print "LENALENALENA", array(lena_parts[0][0])
    #print array(lena_parts[0][0]).shape
    #for x in lena_parts:
    #    plt.imshow(x,cmap='gray')
    #    plt.show()
        
    
    #return 0
     
    for i in range(len(params)):
        for data in lena_parts:
            img=array(data)
            
            print 'Running model %d of %d.' % (i+1, runs)
            opts.append(run(img,*params[i]))
    return opts

def compress(original,basis):
            return dot(original, basis)
def run():
    opts, args = getopt.getopt(sys.argv[1:],"l")
    params = load_params(param_file(args),param_opt(opts))
    
    while params:
        thisParam = [params.pop(0)]

        
        for data in run_model(args[0], thisParam):
            print 'Saving data.'
            save_data('/'.join([out_path, data.label+'/']), data)
        

    print 'Model(s) ran successfully.'
    sys.exit()


if __name__ == '__main__':
    run()
