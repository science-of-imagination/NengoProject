'''
Tools for displaying data.
'''

from matplotlib.pyplot import imshow
from numpy import reshape

def figure_from_vector(vector, canvas_size):
    img = reshape(vector, (canvas_size, canvas_size), 'F')        
    imshow(img.T, cmap='gray')
