'''This module is used to run nengo models. See README file.'''

#out_path determines where model data will be saved. You must create
# cfg.py and set out_path to be the path to the directory you want
# your models to be stored in. You do not need to manually create this
# directory.
from cfg import out_path
from utils.collect import save_data
import sys
import getopt


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
    return [run(*args) for args in params]


def run():
    opts, args = getopt.getopt(sys.argv[1:],"l")    
    for data in run_model(args[0],
                          load_params(param_file(args),
                                      param_opt(opts))):
        save_data('/'.join([out_path, data.label+'/']), data)
    sys.exit()


if __name__ == '__main__':
    run()
