from data import load_data
from utils.visualize import figure_from_vector
from matplotlib.pyplot import show, savefig, clf

path = './model_outputs/feed_forward_gabor_canvas/'
filename = '20150114170247patches_1_4105'
extension = '.pkl.gz'
data = load_data(path+filename+extension)
#for i in range(len(data.data)):
#    figure_from_vector(data.data[i], 28)
#    savefig(path+'/imgs/'+filename+'/'+str(i)+'.png')
#    print 'Saved img '+str(i)
#    clf()
avg = sum(data.data[:100])/len(data.data[0:100])
figure_from_vector(avg, 28)
show()
print 'done'
