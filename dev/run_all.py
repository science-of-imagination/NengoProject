import utils.encoders as enc
from data import load_mini_mnist
from utils.collect import Model
from utils.modelbuilder import modelbuilder

#PROPERTIES
stimulus = 0
#ipts = list(load_mini_mnist('train')[:5][stimulus])
ipts = [1.0, 0.0, 0.0, 1.0]
null = [0 for x in range(len(ipts))]
duration = 0.2#s
printimgs = True
avgimg = True
encoderfuncs = [enc.normalized_random_gabor_encoders, enc.pixel_encoders]

params = {'canvas_size' : 2,
          'ens_size' : 2**2,
          'sigmarng' : [0.05, 0.1],
          'encoderfunc' : encoderfuncs[1],
          'iptfunc' : lambda t: (t>0.2 and null) or ipts,}

#PROGRAM
model = Model(modelbuilder, params, ipts)

model.run(duration, printimgs, avgimg)
