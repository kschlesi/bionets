### TEST script for pybrain network building, training and visualization
#

import numpy as np
import pb_net as pbn
import pb_session as pbs
import networkx as nx
from pybrain.supervised.trainers import BackpropTrainer

# create test network
layers = [1,5,6,4,1]
n = pbn.pbFFNet(layers)

# connect layers Full, Random, Chosen
ci0 = n.connectLayers("in","hidden0",np.ones([1,5]))
c01 = n.connectLayers("hidden0","hidden1",np.random.randint(0,2,[5,6]))
#c01 = n.connectLayers("hidden0","hidden1",np.ones([5,5]))
mm = np.zeros([6,4])
mm[[0,1,2],0] = 1
mm[[3,4,5],1] = 1
mm[[0,1,2],2] = 1
mm[[3,4,5],3] = 1
c12 = n.connectLayers("hidden1","hidden2",mm)
#m = np.array([[1,1],[1,0],[0,1],[1,1]])
m = np.array([[1],[1],[0],[1]])
c2o = n.connectLayers("hidden2","out",m)

# display layers & plot
n.dispLayers()
#n.dispNet()
n.netPlot("Layers")

# create dataset from ann's data
dset = pbs.createDS(pbs.annDS())
print(n.params)
n.params[1] = 0
print(n.params)

# create and utilize backprop trainer
trainer = BackpropTrainer(n,dset)
trainer.trainEpochs(100)
