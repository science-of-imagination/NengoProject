
import numpy
import random
from PIL import Image
from scipy.sparse import csc_matrix
from sparsesvd import sparsesvd

def gabor(size, lambd, theta, psi, sigma, gamma, x_offset, y_offset):
  x = numpy.linspace(-1, 1, size)
  y = numpy.linspace(-1, 1, size)
  X, Y = numpy.meshgrid(x, y)
  X = X - x_offset
  Y = Y + y_offset

  cosTheta = numpy.cos(theta)
  sinTheta = numpy.sin(theta)
  xTheta = X * cosTheta  + Y * sinTheta
  yTheta = -X * sinTheta + Y * cosTheta
  e = numpy.exp( -(xTheta**2 + yTheta**2 * gamma**2) / (2 * sigma**2) )
  cos = numpy.cos(2 * numpy.pi * xTheta / lambd + psi)
  return e * cos

def makerandomgabor(size):
  sigma = random.uniform(0.1, 0.2)
  #Choice of r makes gabors stay within half width of center. 
  #Also, squaring ensures gabors are more frequent near center.
  r = (random.uniform(0,1)-sigma)**2
  th = random.uniform(0, 2*numpy.pi)
  return gabor(size, 
                lambd=random.uniform(0.3, 0.8),
                theta=random.uniform(0, 2*numpy.pi),
                psi=random.uniform(0, 2*numpy.pi),
                sigma=sigma,
                gamma=random.uniform(0.7, 1),
                x_offset=r*numpy.cos(th),
                y_offset=r*numpy.sin(th))

def converttoarray(file_, scalew, scaleh):
  """Converts a picture in file_ to a compressed, grayscaled flattened array."""
  obj=Image.open(file_)
  obj=obj.convert('LA').resize((scalew, scaleh), Image.NEAREST)
  array=numpy.array(list(obj.getdata(band=0)),float)
  array.shape=(obj.size[1],obj.size[0])
  matrix=numpy.matrix(array)
  flatten=numpy.array(matrix.flatten('F'))[0]
  flatten/=255.0
  return flatten

def setupencodersandimages(listoffiles, scalew, scaleh, numgabor):
  combined=[]
  for file_ in listoffiles:
    combined.append(converttoarray(file_, scalew, scaleh))
  imgs=numpy.array([img/numpy.linalg.norm(img) for img in combined]).T
  csc=csc_matrix(imgs)

  ut, S, vt=sparsesvd(csc, len(listoffiles))
  M=numpy.diag([numpy.linalg.norm(ut[i:]) for i in range(ut.shape[0])])
  #W is M inverse
  UW=numpy.dot(ut.T, numpy.linalg.inv(M))
  MSvt=numpy.dot(M, numpy.dot(numpy.diag(S), vt))
  gaborenc=[makerandomgabor(scalew*scaleh) for i in range(numgabor)]
  gaborenc=[(1/numpy.linalg.norm(i).flatten())*i for i in gaborenc]
  return imgs, UW, MSvt, gaborenc

def setupnengo(stepsperinput, numgabor, listoffiles, scalew, scaleh):
  import nef
  imgs, UW, MSvt, gaborenc=setupencodersandimages(listoffiles, scalew, scaleh,
                                                  numgabor)
  dt=0.001
  dims=len(listoffiles)

  class Input(nef.SimpleNode):
    def origin_input(self, dimensions=scalew*scaleh):
      step=int(round(self.t/dt))
      index=(step/stepsperinput)%len(listoffiles)
      return imgs[index]

  class Output(nef.SimpleNode):
    def termination_save(self, x, dimensions=scalew*scaleh, pstc=0.01):
      step=int(round(self.t/dt))
      index=(step/stepsperinput)%len(listoffiles)
      out=[str(step),listoffiles[index]]
      out.extend([str(val) for val in x])
      if step%stepsperinput==(stepsperinput-1):
        f=file('K:\\experiment.csv', 'a+')
        f.write(','.join(out))
        f.write('\n')
        f.close()

  net=nef.Network('VisionNet')
  input_=net.add(Input('input'))
  output=net.add(Output('output'))
  net.make('renderer', scalew*scaleh, dims, encoders=UW)
  net.make('ibuffer', numgabor, scalew*scaleh,
           encoders=numpy.array([encoder.flatten() for encoder in gaborenc]))
  net.connect(input_.getOrigin('input'), 'renderer')
  net.connect('renderer', 'ibuffer', function=lambda v: numpy.dot(UW, v),
              synapse=0.01)
  net.connect('ibuffer', output.getTermination('save'),
              function=lambda x: 0.01*x, synapse=0.01)
  
  sim=nef.Simulator(net)
  sim.run(1)

LoF=['H:\\Git\\NengoProject\\Pictures\\circle.png',
     'H:\\Git\\NengoProject\\Pictures\\square.png']
setupnengo(100, 700, LoF, 200, 200)
