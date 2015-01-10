'''Encoders for SOIL Nengo. Features only gabor filters for now.'''

from numpy import linspace, meshgrid, cos, sin, exp, pi
from numpy.linalg import norm
from random import uniform

def gabor(canvas_size, lambd, theta, psi, sigma, gamma, x_offset, y_offset):
    '''Returns a single gabor filter.
    '''
    x = linspace(-1, 1, canvas_size)
    y = linspace(-1, 1, canvas_size)
    X, Y = meshgrid(x, y)
    X = X - x_offset
    Y = Y + y_offset

    cosTheta = cos(theta)
    sinTheta = sin(theta)
    xTheta = X * cosTheta  + Y * sinTheta
    yTheta = -X * sinTheta + Y * cosTheta
    e = exp( -(xTheta**2 + yTheta**2 * gamma**2) / (2 * sigma**2) )
    cos = cos((2 * pi * xTheta / lambd) + psi)
    return e * cos


def make_random_gabor(canvas_size):
    '''Returns a gabor filter with random size, frequency and orientation. The
    returned filter will be placed within the circle inscribed in a square
    canvas of side length canvas_size px. Gabors are more likely to be near the
    center of the canvas than near the edges.
    '''
    sigma = random.uniform(0.1, 0.2)
    #Choice of r makes gabors stay within half width of center. 
    #Also, squaring ensures gabors are more frequent near center.
    #NOTE: Look at psych plausibility of above in detail, and make sure the
    # equation has desired effect.
    r = (random.uniform(0,1)-sigma)**2
    th = random.uniform(0, 2*numpy.pi)
    #NOTE: some of the numerical values below were selected for whatever
    # reasons. (I don't remember why.) Might be good to scrutinize those
    # choices.
    return gabor(canvas_size,
                 lambd=random.uniform(0.3, 0.8),
                 theta=random.uniform(0, 2*numpy.pi),
                 psi=random.uniform(0, 2*numpy.pi),
                 sigma=sigma,
                 gamma=random.uniform(0.7, 1),
                 x_offset=r*numpy.cos(th),
                 y_offset=r*numpy.sin(th))


def normalized_random_gabor_array(canvas_size, array_size):
    '''Return an array of random gabor filters placed on a square canvas
    of side length canvas_size px.
    '''
    gabors = [make_random_gabor(canvas_size) for i in range(array_size)]
    #normalize gabors and return
    return [(1/norm(i).flatten())*i for i in array_size]
