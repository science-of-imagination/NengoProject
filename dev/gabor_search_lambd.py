from numpy import dot,pi,reshape
from utils.encoders_search import normalized_random_gabor_encoders,pixel_encoders
from data import load_img
import matplotlib.pyplot as plt




def stimulus_through_encs(enc,stimulus):
    return dot(dot(encs,stimulus),encs)

canvas = 100
N = 20000
#theta:pi/3,2*pi/3
#thetas = ['0','pi/6','pi/4','pi/3','pi/2','2*pi/3','3*pi/4','5*pi/6','pi','7*pi/6','5*pi/4','4*pi/3','3*pi/2','5*pi/3','7*pi/4','11*pi/6']
div = repr(float(N))
img = load_img('data/lena_512x512.png',(canvas,canvas))
r='uniform(2.0/canvas_size,1.4)'
theta='uniform(pi/3,2*pi/3)'
lambd='lognormal(0.001,0.0005)'
sigma='0.01'#'0.01'
gamma='sigma*24'
th='uniform(0,2*pi)'
psi='uniform(0,pi/2)'
for i in range(1,5):
    #gamma = 0 + (i*2/100.0)
    #sigma = 0 + (i/100.0)   
    #lambd_end = 0 + (i/1000.0)
    #lambd = 'uniform(2/canvas_size,' + repr(lambd_end) + ')'
    encs = normalized_random_gabor_encoders(canvas, N,thetaSTR=theta,lambdSTR=lambd,sigmaSTR=sigma,gammaSTR=repr(1.0),x_offSTR='uniform(-1,1)',
                                            y_offSTR='uniform(-1,1)',psiSTR=psi,thSTR=th,rSTR=r)
    
    output = stimulus_through_encs(encs,img)
    output = output#/max(output)
    print max(output)    
    out_img = reshape(output, (canvas,canvas), 'F')
    #print(out_img.shape)
    #plt.imsave('/home/sterling/NewNengoOutput/all/'+'N'+repr(N)+'lambd'+lambd+'sigma'+repr(sigma)+'theta'+theta+'gamma'+repr(gamma)+'_'+repr(i)+'.png',
     #           out_img.T, cmap='gray') 
    plt.imsave('/home/sterling/NewNengoOutput/all/justoutput_' + repr(i+20) + '.png', out_img.T, cmap='gray') 
  
    mse = ((img-output.T) ** 2).mean()
       
    print theta,lambd,mse

#for start in thetas:
#    theta = ''
#    
#    #print theta, ' one'     
#    for end in thetas:
#        theta = 'uniform('+start
#        theta = theta+','+end+')'
#        #print theta, ' two'
#        encs = normalized_random_gabor_encoders(canvas, N,thetaSTR=theta)
#        output = stimulus_through_encs(encs,img)#

#        mse = ((img-output) ** 2).mean()
        
#        print theta,mse

