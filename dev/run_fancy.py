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
    #for i in range(len(params)):
    ruwn = 0
    for p,data in gratings:
        img=data
        #print params
        print 'Stimulus %d of %d.' % (ruwn+1, runs)
        opts.append(run(img,p,*params))
        ruwn +=1
    return opts


def run():
    opts, args = getopt.getopt(sys.argv[1:],"l")
    params = load_params(param_file(args),param_opt(opts))
    #params1 = params[:len(params)/2]
    #params2 = params[len(params)/2:]
    #split the running in 2, save data half way through
    moedels =len(params)
    moedel = 0
    while params:
        thisParam = params.pop()
        print 'Running model %d of %d' % (moedel+1, moedels)
        moedel +=1
        for data in run_model(args[0], thisParam):
            print 'Saving data.'
            save_data('/'.join([out_path, data.label+'/']), data)
        

    print 'Model(s) ran successfully.'
    sys.exit()


if __name__ == '__main__':
    run()