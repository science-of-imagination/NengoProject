from tkFileDialog import askopenfilenames
from pylab import plot, show
from data import load_data

queue = askopenfilenames()
if isinstance(queue, unicode):
    queue = queue.encode('ascii', 'replace').split()
data = []
for path in queue:
    data.append(load_data(path).rmses[49])
plot(data)
show()
 
