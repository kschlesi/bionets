### BIONETS NEURAL NETWORKS using PYBRAIN

# This code defines a class of FF/BP neural network for the bionets project.
# Network properties include an arbitrary number of hidden layers with equal
# number of nodes in each, an input and output layer with bias nodes, the
# ability to individually define layer-to-layer connections.
# The network is trained with:
# The network can be dynamically attacked or degraded by removing nodes in
# between individual training sessions.

import math
import numpy as np
import random
import matplotlib as mpl
import scipy as sp
import pybrain as pb

class pbFFNet:

    def __init__(self, layers, nName="net", layerTypes=["lin","sig","sig"]):
        # layers = list with number of nodes per layer
        # no. layers= len(layers), no. nodes= sum(layers)
        # layers(1) = input layer, layers(end) = output layer
        self.layers = layers
        self.nHL = len(layers)-2
        self.nNodes = sum(layers)

        # initialize and name all nodes
        # to add: ability to change transfer functions within layer nodes
        self.net = pb.structure.FeedForwardNetwork(name=nName)
        self.net.addInputModule(pb.structure.LinearLayer(self.layers[0], name="in"))
        self.net.addOutputModule(pb.structure.SigmoidLayer(self.layers[self.nHL+1], name="out"))
        for i in range(self.nHL):
            lName = "hidden" + str(i)
            self.net.addModule(pb.structure.SigmoidLayer(self.layers[i+1], name=lName))

    def connectLayers(self,L1,L2,m):
        # L1 = string (name of upstream layer); L2 = string (name of downstream layer)
        # m = binary matrix of size(#nodes(L1),#nodes(L2))
        if np.all(m):
            cName = L1 + "_" + L2 + "_F"
            fcxn = pb.structure.FullConnection(self.net[L1], self.net[L2], name=cName)
            self.net.addConnection(fcxn)
        else:
            for i in range(np.shape(m)[0]):
                for j in range(np.shape(m)[1]):
                    if m[i,j]==1:
                        cName = L1 + "-" + str(i) + "_" + L2 + "-" +str(j)
                        cxn = pb.structure.connections.connection.Connection(self.net[L1], self.net[L2], \
                                                      name=cName, \
                                                      inSliceFrom=i, inSliceTo=i+1, \
                                                      outSliceFrom=j, outSliceTo=j+1)
                        self.net.addConnection(cxn)

    def dispLayers(self):
        self.net.sortModules()
        return self.net.modulesSorted

    def dispNet(self):
        return self.net.connections

    #def removeNode(self,lName,node):
        # remove a given node from the network while keeping existing params

    #def feedForward(self,x):
        # train on a dataset
