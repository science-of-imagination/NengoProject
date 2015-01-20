from tkFileDialog import askopenfilenames
from utils.visualize import mk_imgs
from data import load_data

queue = askopenfilenames().split()
for path in queue:
    mk_imgs(path.strip('.pkl.gz')+'/', load_data(path))
