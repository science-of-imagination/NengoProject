from numpy import dot,pi
from utils.encoders_search import normalized_random_gabor_encoders,pixel_encoders
from data import load_img





def stimulus_through_encs(enc,stimulus):
    return dot(dot(encs,stimulus),encs)

canvas = 100
N = 5000
thetas = ['0','pi/6','pi/4','pi/3','pi/2','2*pi/3','3*pi/4','5*pi/6','pi','7*pi/6','5*pi/4','4*pi/3','3*pi/2','5*pi/3','7*pi/4','11*pi/6']
img = load_img('data/lena_512x512.png',(canvas,canvas))
for start in thetas:
    theta = ''
    
    #print theta, ' one'     
    for end in thetas:
        theta = 'uniform('+start
        theta = theta+','+end+')'
        #print theta, ' two'
        encs = normalized_random_gabor_encoders(canvas, N,thetaSTR=theta)
        output = stimulus_through_encs(encs,img)

        mse = ((img-output) ** 2).mean()
        
        print theta,mse

