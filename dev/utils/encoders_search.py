'''Encoders for SOIL Nengo. Features only gabor filters for now.'''

from numpy import array, linspace, meshgrid, cos, sin, exp, pi, identity, log
from numpy.linalg import norm
from random import uniform, choice

def pixel_encoders(canvas_size):
    return identity(canvas_size**2)

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

def make_random_gabor(canvas_size,thSTR,rSTR,lambdSTR,thetaSTR,psiSTR,sigmaSTR,gammaSTR,x_offSTR,y_offSTR):
    '''Returns a gabor filter with random size, frequency and orientation. The
    returned filter will be placed within the circle inscribed in a square
    canvas of side length canvas_size px. Gabors are more likely to be near the
    center of the canvas than near the edges.
    '''
    
    #th = uniform(0, 2*pi)
    #r = uniform(2.0/canvas_size,1.4)
    exec('th='+thSTR)
    exec('r='+rSTR)
    exec('lambd_s='+lambdSTR)
    exec('theta_s='+thetaSTR)
    exec('psi_s='+psiSTR)
    exec('sigma_s='+sigmaSTR)
    exec('gamma_s='+gammaSTR)
    exec('x_offset_s='+x_offSTR)
    exec('y_offset_s='+y_offSTR)

    return gabor(canvas_size,
                 lambd=lambd_s,#uniform(2.0/canvas_size, 1),
                 theta=theta_s,#uniform(0, 2*pi),
                 psi=psi_s,#uniform(0, 2*pi),
                 sigma=sigma_s,#log(0.1*r+1),
                 gamma=gamma_s,#0.24,#uniform(0.07, 1),
                 x_offset=x_offset_s,#r*cos(th),
                 y_offset=y_offset_s)#r*sin(th))


def normalized_random_gabor_encoders(canvas_size, array_size,thSTR='uniform(0, 2*pi)',
                    rSTR='uniform(2.0/canvas_size,1.4)',
                    lambdSTR='uniform(2.0/canvas_size,1)',
                    thetaSTR='uniform(0,2*pi)',
                    psiSTR='uniform(0,2*pi)',
                    sigmaSTR='log(0.1*r+1)',
                    gammaSTR='0.24',
                    x_offSTR='r*cos(th)',
                    y_offSTR='r*sin(th)'):
    '''Return an array of random gabor filters placed on a square canvas
    of side length canvas_size px.
    '''
    gabors = [make_random_gabor(canvas_size,thSTR,rSTR,lambdSTR,thetaSTR,psiSTR,sigmaSTR,gammaSTR,x_offSTR,y_offSTR).flatten()
              for i in range(array_size)]
    #normalize gabors and return
    return array([(1/norm(i).flatten())*i for i in gabors])    
