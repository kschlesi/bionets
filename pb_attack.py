### BIONETS NEURAL NETWORKS using PYBRAIN

# This code defines a class of FF/BP neural network for the bionets project.
# Network properties include an arbitrary number of hidden layers and nodes
# per layer, an input and output layer with bias nodes, the
# ability to individually define layer-to-layer connections.
# the network can be visualized with networkx.
# The network can separately be trained with a pybrain BackPropTrainer.
# The network can be dynamically attacked or degraded by removing nodes in
# between individual training sessions.

import math
import numpy as np
import random
#import matplotlib.pyplot as pp
#import scipy as sp
#from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
#import networkx as nx
from pb_net import pbFFNet
from maskedconnections import FullMaskedConnection

class pbFFAttackNet(pbFFNet):
    # inherits from pbFFNet and adds the ability to remove nodes.

    def __init__(self, layers, attacks, nName="net", layerTypes=["lin","sig","sig"]):
        # initialize pbFFNet
        pbFFNet.__init__(self, layers, nName, layerTypes)
        # attacks = dict of times(epochs) to attack, and nodes to be attacked
        self.attacks = attacks

    def connectLayers(self,L1,L2,m):
        """override connectLayers to use MASKED layers"""
        # L1 = string (name of upstream layer); L2 = string (name of downstream layer)
        # m = binary matrix of size(#nodes(L1),#nodes(L2))

        # create a full connection between layers
        if np.all(m):
            cName = L1 + "_" + L2 + "_F"
            cxns = FullMaskedConnection(self[L1], self[L2], name=cName)
            self.addConnection(cxns)
            cxns = [cxns]
        # create individual 'full' connections between individual node pairs in the layers
        else:
            cxns = []
            for i in range(np.shape(m)[0]):
                for j in range(np.shape(m)[1]):
                    if m[i,j]==1:
                        cName = L1 + "-" + str(i) + "_" + L2 + "-" + str(j)
                        cxn = FullMaskedConnection(self[L1], self[L2], \
                                             name=cName, \
                                             inSliceFrom=i, inSliceTo=i+1, \
                                             outSliceFrom=j, outSliceTo=j+1)
                        self.addConnection(cxn)
                        cxns = cxns + [cxn]
        # save m connectivity matrix in self.layerConns
        if L1=="in":
            lCindex = 0;
        elif L2=="out":
            lCindex = self.nHL
        else:
            lCindex = int(L1[6:len(L1)])+1
        self.layerConns[lCindex] = m
        return cxns


    #def removeNode(self,lName,node):
        # remove a given node from the network while keeping existing params

    def removeParam(self,lName,node):
        # remove a given param from the network by masking in its connection
        self[lName].maskParam(node)
        print(self[lName].params)
