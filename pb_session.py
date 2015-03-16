#### class of network sessions containing methods for
# creating datasets and networks and training networks
# on those datasets

import numpy as np
import scipy as sp
import random
import pb_net as pbn
from pybrain.datasets import SupervisedDataSet

class pbNetSession:

    def __init__(self,layers,x,dimIn=1,dimTar=1):
        self.n = []
        self.ds = []
        createDS(x, dimIn, dimTar)

    #def trainFFBP(self,iter):


######################################################################

def annFnc(x):
    return .1*np.sin(10*x)**2+.4*np.sin(30*x)**2+0.5*np.sin(18*x)**2

def annDS():
    points = np.array([
          [.1,annFnc(.1)],
          [.26,annFnc(.26)],
          [.42,annFnc(.42)],
          [.58,annFnc(.58)],
          [.74,annFnc(.74)],
          [.9,annFnc(.9)],
        ])
    return points

def createDS(x,dimIn=1,dimTar=1):
    # create a pybrain dataset
    [ns,d] = x.shape
    assert d == dimIn + dimTar
    from pybrain.datasets import SupervisedDataSet
    ds = SupervisedDataSet(dimIn,dimTar)
    for inp, tar in zip(x[:,0:dimIn], x[:,dimIn:d]):
        #print(inp[:])
        #print(tar[:])
        ds.addSample(inp,tar)
    return ds

########################################################################
