'''Encoders for SOIL Nengo. Features only gabor filters for now.'''

from numpy import array, linspace, meshgrid, cos, sin, exp, pi, identity
from numpy.linalg import norm
from random import uniform, choice

def pixel_encoders(canvas_size):
    return identity(canvas_size**2)

def patch(canvas_size, radius, x_offset, y_offset, sign):
    x = linspace(-1, 1, canvas_size)
    y = linspace(-1, 1, canvas_size)
    X, Y = meshgrid(x, y)
    X = X - x_offset
    Y = Y + y_offset

    cosTheta = cos(0)
    sinTheta = sin(0)
    xTheta = X * cosTheta  + Y * sinTheta
    yTheta = -X * sinTheta + Y * cosTheta
    e = exp( -(xTheta**2 + yTheta**2) / (2 *radius**2) )    
    return sign*e

def make_random_patch(canvas_size):
    sigma = uniform(0.05, 1)    
    r = (uniform(0,1)-sigma)**2
    th = uniform(0, 2*pi) 
    sign = choice([-1,1])
    return patch(canvas_size, sigma, r*cos(th), r*sin(th), sign)

def normalized_random_patch_encoders(canvas_size, array_size):
    '''Return an array of random gabor filters placed on a square canvas
    of side length canvas_size px.
    '''
    gabors = [make_random_patch(canvas_size).flatten()
              for i in range(array_size)]
    #normalize patches and return
    return array([(1/norm(i).flatten())*i for i in gabors])    

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
    cosed = cos((2 * pi * xTheta / lambd) + psi)
    return e * cosed


def make_random_gabor(canvas_size):
    '''Returns a gabor filter with random size, frequency and orientation. The
    returned filter will be placed within the circle inscribed in a square
    canvas of side length canvas_size px. Gabors are more likely to be near the
    center of the canvas than near the edges.
    '''
    sigma = uniform(0.05, 0.1)
    #Choice of r makes gabors stay within half width of center. 
    #Also, squaring ensures gabors are more frequent near center.
    #NOTE: Look at psych plausibility of above in detail, and make sure the
    # equation has desired effect.
    r = (uniform(0,1)-sigma)**2    
    #r = uniform(0,0.8)
    th = uniform(0, 2*pi)
    #NOTE: some of the numerical values below were selected for whatever
    # reasons. (I don't remember why.) Might be good to scrutinize those
    # choices.
    return gabor(canvas_size,
                 lambd=uniform(0.3, 0.8),
                 theta=uniform(0, 2*pi),
                 psi=uniform(0, 2*pi),
                 sigma=sigma,
                 gamma=uniform(0.7, 1),
                 x_offset=r*cos(th),
                 y_offset=r*sin(th))


def normalized_random_gabor_encoders(canvas_size, array_size):
    '''Return an array of random gabor filters placed on a square canvas
    of side length canvas_size px.
    '''
    gabors = [make_random_gabor(canvas_size).flatten()
              for i in range(array_size)]
    #normalize gabors and return
    return array([(1/norm(i).flatten())*i for i in gabors])    
