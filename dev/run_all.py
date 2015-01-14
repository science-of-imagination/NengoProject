from data import load_mini_mnist
from utils.collect import Model
from utils.modelbuilder import modelbuilder

#PROPERTIES
stimulus = 0
ipts = list(load_mini_mnist('train')[:5][stimulus])
null = [0 for x in range(len(ipts))]
duration = 0.5#s

params = {'canvas_size' : 28,
          'ens_size' : 200,
          'sigmarng' : [0.05, 0.1],
          'iptfunc' : lambda t: (t>0.2 and null) or ipts,}

#PROGRAM
model = Model(modelbuilder, params, ipts)

model.run(duration)
