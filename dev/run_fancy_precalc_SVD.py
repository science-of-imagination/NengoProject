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
from numpy.linalg import norm, pinv
from numpy import array, ones, sqrt, subtract, dot, where, zeros, flipud
from utils.encoders import mk_bgbrs, normalized_random_gabor_encoders, mk_gbr_eval_pts

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


def run_model(model_name, params):
    exec 'from models.%s import run' % model_name
    opts = []
    gratings = load_data('./data/gratings.pkl')
    runs = len(gratings)
    #print params, "PARAAMAMAMDMAFMADFAD"
    #return 0
    
    ruwn = 0
    for p,data in gratings:
        img=data
        #print params
        print 'Stimulus %d of %d.' % (ruwn+1, runs)
        opts.append(run(img,p,*params))
        ruwn +=1
    return opts

def compress(original,basis):
    return dot(original, basis)

def uncompress(compressed):
    return dot(basis, compressed.T).T

def run():
    opts, args = getopt.getopt(sys.argv[1:],"l")
    #params = load_params(param_file(args),param_opt(opts))

    params = load_params(param_file(args),param_opt(opts))
    num_params = len(params)
    count = 0
    while params:
        count += 1
        thisParam = [params.pop(0)]
        N, n_eval_pts, w, h  = int(thisParam[0][0]), int(thisParam[0][1]), int(thisParam[0][2]), int(thisParam[0][3])
        dims = (w,h)
                
        print ' Initializing encoders.'
        encs = array(mk_bgbrs(N/2, dims, 4))
        print 'Initializing eval points.'
        eval_points = mk_gbr_eval_pts(n_eval_pts, dims[0])

        print 'Initializing SVD compression.'
        U, S, V = svds(encs.T, 600)
        S=flipud(S)
        #import pylab
        #pylab.plot(S)
        #pylab.show()
        #print where(S<S[0]*0.01)
        D = where(S<S[0]*0.01)[0][0]
        print D
        basis = array(U[:,-D:])

        comp_encs = compress(encs)
        comp_decs = compress(eval_points)
        del encs
        del decs

        
        newParams = [comp_encs,comp_decs,basis,D] + thisParam[0]
        
        print 'Running model %d of %d' % (count, num_params)
        
        for data in run_model(args[0], newParams):
            print 'Saving data.'
            save_data('/'.join([out_path, data.label+'/']), data)
        

    print 'Model(s) ran successfully.'
    sys.exit()


if __name__ == '__main__':
    run()
