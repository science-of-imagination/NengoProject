import nengo

model = nengo.Network(label="My Network")
with model:
    input = nengo.Node([0, 0])

    a = nengo.Ensemble(100, dimensions=2, label="Image Memory")
    b = nengo.Ensemble(100, dimensions=2, label="Output")
    nengo.Connection(a,b)
    nengo.Connection(b,b)
    nengo.Probe(a)
    nengo.Probe(a.neurons)

    nengo.Connection(input, a)

import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 3.0117143482592477
gui[model].offset = -201.23714786073788,-319.34286965184936
gui[a].pos = 214.335, 173.105
gui[a].scale = 1.000
gui[b].pos = 299.943, 199.668
gui[b].scale = 1.000
gui[input].pos = 98.672, 153.183
gui[input].scale = 1.000
