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

class pbFFAttackNet(pbFFNet):
    # inherits from pbFFNet and adds the ability to remove nodes.

    def __init__(self,layers,attacks,name=nName,layerTypes=["lin","sig","sig"]):
        # initialize pbFFNet
        pbFFNet.__init__(self, layers, nName, layerTypes)
        # attacks = dict of times(epochs) to attack, and nodes to be attacked
        self.attacks = attacks

    def removeNode(self,lName,node):
        # remove a given node from the network while keeping existing params
        # save old connections
        oldConnsOut = self.connections[self[lName]].copy()
        if lName=="in":
            oldConnsIn = []
        else:
            oldConnsIn = self.connections[self[self._prelayer(lName)]].copy()



    def _prelayer(self,lName):
        if lName = "in":
            prelayer = []
        elif lName=="out":
            prelayer = "hidden" + str(self.nHL-1)
        elif lname=="hidden0"
            prelayer = "in"
        else:
            prelayer = "hidden" + str(int(lName[6:len(lName)])-1)
        return prelayer
