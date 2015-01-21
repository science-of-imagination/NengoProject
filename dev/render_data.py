from tkFileDialog import askopenfilenames
from utils.visualize import mk_imgs, mk_plt_imgs
from data import load_data

queue = askopenfilenames()
if isinstance(queue, str):
    queue = queue.split()
for path in queue:
    mk_plt_imgs(path.strip('.pkl.gz')+'/', load_data(path))
