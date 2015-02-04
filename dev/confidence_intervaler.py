from tkFileDialog import askopenfilenames
from utils.visualize import mk_imgs, mk_plt_imgs
from utils.collect import cos_simi
from data import load_data

import numpy as np
import scipy as sp
import scipy.stats


def collapse(data, step=0, axis=0):
    '''Collapse together every step columns.
    
    Parameters:
    
        data: array of numerical data.

        step: int'''
    #arrs = np.split(data, step, axis)
    #arrs = [np.mean(chunk, axis) for chunk in arrs]
    #arrs = np.concatenate(arrs, axis)
    #return arrs 
    
    arr = []
    #if step == 0:
    #    data[]
    for row in range(len(data)):
        new_r = [np.mean(data[row, step*i:step*i+step]) for i in range(row/step)]
        arr.append(new_r)
    return arr
        
def collapse2(data, maxColumns,numGroups):
    data.shape = (15,7)#assuming the right shape....
    #print "data", data
    columnCut = np.delete(data,np.s_[maxColumns:],1)
    #print "columnCut", columnCut
    grouped = np.array_split(columnCut,numGroups,0)
    #print "grouped", grouped
    return grouped
    #return [np.mean(n) for n in grouped]
        

    

    

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t.ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


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
    #dataz.append(data.rmses)
    dataz.append(cos_simi(data.stimulus, data.data[98]))
    #print(cos_simi(data.stimulus, data.data[0]))
    #print data.rmses, "ramsies"
    #print data.data, "data"
    #print data.weights, "weights"
    accu +=1
print "Collapsing..."
groups = collapse2(np.array(dataz),5,5)
CIs = []
for group in groups:
    CIs.append(mean_confidence_interval(group.flatten()))
    
with open('opt.txt', 'wb') as f:
    lines = ''
    for i in range(15):
        #print(dataz[i*7:i*7+7])
        lines += ' '.join([str(x) for x in dataz[i*7:i*7+7]]) + '\n'
    
    #f.write(lines)

    for i in CIs:
        lines += ' '.join(str(i[0]),str(i[1]),str(i[2])) + '\n'
    f.write(lines)


