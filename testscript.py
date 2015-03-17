### TEST script for pybrain network building, training and visualization
#

import numpy as np
import pb_net as pbn
import pb_session as pbs
import networkx as nx
from pybrain.supervised.trainers import BackpropTrainer

# create test network
layers = [1,5,5,1]
n = pbn.pbFFNet(layers)

# connect layers Full, Random, Chosen
ci0 = n.connectLayers("in","hidden0",np.ones([1,5]))
c01 = n.connectLayers("hidden0","hidden1",np.random.randint(0,2,[5,5]))
#c01 = n.connectLayers("hidden0","hidden1",np.ones([5,5]))
#mm = np.zeros([5,1])
#mm[[0,1,2],0] = 1
#mm[[3,4,5],1] = 1
c1o = n.connectLayers("hidden1","out",np.ones([5,1]))

# display layers
n.dispLayers()
n.dispNet()

# create dataset from ann's data
dset = pbs.createDS(pbs.annDS())

# create and utilize backprop trainer
#trainer = BackpropTrainer(n.net,dset)
#trainer.trainEpochs(1000)
