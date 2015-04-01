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
import matplotlib.pyplot as pp
import scipy as sp
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
import networkx as nx
from pb_net import pbFFNet
from maskedlayers import LinearMaskedLayer, SigmoidMaskedLayer

class pbFFAttackNet(pbFFNet):
    # inherits from pbFFNet and adds the ability to remove nodes.

    def __init__(self, layers, attacks, name="net", layerTypes=["lin","sig","sig"]):
        # initialize pbFFNet
        FeedForwardNetwork.__init__(self,name)
        # attacks = dict of times(epochs) to attack, and nodes to be attacked
        self.attacks = attacks
        # layers = list with number of nodes per layer
        # no. layers= len(layers), no. nodes= sum(layers)
        # layers(1) = input layer, layers(end) = output layer
        self.layers = layers
        self.nHL = len(layers)-2
        self.nNodes = sum(layers)
        self.layerConns = [np.zeros([1,1])]*(self.nHL+1)

        # initialize and name all nodes (default: input linear, others sigmoid)
        # to add: ability to change transfer functions within layer nodes
        self.addInputModule(LinearMaskedLayer(self.layers[0], "in"))
        self.addOutputModule(SigmoidMaskedLayer(self.layers[self.nHL+1], "out"))
        for i in range(self.nHL):
            lName = "hidden" + str(i)
            self.addModule(SigmoidMaskedLayer(self.layers[i+1], lName))

    #def removeNode(self,lName,node):
        # remove a given node from the network while keeping existing params

    def removeParam(self,lName,node):
        # remove a given param from the network while keeping existing params
        self[lName].maskParam(node)
        print(self[lName].params)
