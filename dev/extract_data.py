from tkFileDialog import askopenfilenames
from utils.visualize import mk_imgs, mk_plt_imgs
from data import load_data

queue = askopenfilenames()
if isinstance(queue, unicode):
    queue = queue.encode('ascii', 'replace').split()
queue = list(queue)
queue.sort()
dataz = []
accu = 0
le = len(queue)
for path in queue:
    #print 'Reading file %d of %d' % (accu, le)
    data = load_data(path)
    dataz.append(data.rmses)
    #print data.rmses, "ramsies"
    #print data.data, "data"
    #print data.weights, "weights"
    accu +=1
    
with open('opt.txt', 'wb') as f:
    lines = ''
    for i in range(15):
        #print(dataz[i*7:i*7+7])
        lines += ' '.join([str(x) for x in dataz[i*7:i*7+7]]) + '\n'
    f.write(lines)

#with open('opt.txt', 'wb') as f:
#    lines = ''
#    for k in range(2):
#        for i in range(3):
#            for j in range(7):
#                lines += str(dataz[k*21+i*7+j])+' '
#            lines += '\n'
#    f.write(lines)
