from tkFileDialog import askopenfilenames
from utils.visualize import mk_imgs, mk_plt_imgs
from utils.collect import cos_simi
from data import load_data

from matplotlib.pyplot import imshow, savefig, axis, figure, close, imsave

import numpy as np
import pylab
def plot_gabors(data, columns=None):
    if columns is None:
        columns = int(np.sqrt(len(data)))
    pylab.figure(figsize=(10,10))
    vmin = np.min(data)
    vmax = np.max(data)
    width = int(np.sqrt(data.shape[1]))
    for i, d in enumerate(data):
        w = columns - 1 - (i % columns)
        h = i / columns
        d.shape = width, width
        pylab.imshow(d, extent=(w+0.025, w+0.975, h+0.025, h+0.975),
            interpolation='none', vmin=vmin, vmax=vmax, cmap='gray')
        pylab.xticks([])
        pylab.yticks([])
    pylab.xlim((0, columns))
    pylab.ylim((0, len(data) / columns))


queue = askopenfilenames()
if isinstance(queue, unicode):
    queue = queue.encode('ascii', 'replace').split()
queue = list(queue)
queue.sort()
dataz = []
accu = 0
le = len(queue)
for path in queue:
    print 'Reading file %d of %d' % (accu, le)
    data = load_data(path)
    dataz.append(data.data[49].T)
    #dataz.append(cos_simi(data.stimulus, data.data[49]))
    #print data.rmses[98]
    #plot_gabors(np.array([data.stimulus, data.data[98]]))
    #accu +=1
    #pylab.show()    
plot_gabors(np.array(dataz),32)
pylab.show()  
#count = 0  
#for data in dataz:
#    print data.shape
#    name = './Lenas/' + repr(count) + '.png'
#    imsave(name, data.T, cmap='gray')
    

#with open('opt.txt', 'wb') as f:
#    lines = ''
#    for k in range(2):
#        for i in range(3):
#            for j in range(7):
#                lines += str(dataz[k*21+i*7+j])+' '
#            lines += '\n'
#    f.write(lines)
