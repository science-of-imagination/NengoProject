from tkFileDialog import askopenfilenames
from utils.visualize import mk_imgs, mk_plt_imgs
from data import load_data

queue = askopenfilenames()
if isinstance(queue, unicode):
    queue = queue.encode('ascii', 'replace').split()
for path in queue:
    data = load_data(path)
    #mk_imgs(path.strip('.pkl.gz')+'/', load_data(path))
    mk_plt_imgs(path.strip('.pkl.gz')+'/', load_data(path))
