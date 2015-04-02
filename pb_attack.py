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
from pb_net import pbFFNet, _convParamToNode, _convNodeToParam, _nxNN
from maskedconnections import FullMaskedConnection
import networkx as nx

class pbFFAttackNet(pbFFNet):
    # inherits from pbFFNet and adds the ability to remove nodes.

    def __init__(self, layers, attacks, nName="net", layerTypes=["lin","sig","sig"]):
        # initialize pbFFNet
        pbFFNet.__init__(self, layers, nName, layerTypes)
        # attacks = dict of times(epochs) to attack, and nodes to be attacked
        self.attacks = attacks

    def connectLayers(self, L1, L2, m):
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

    def removeParam(self, iNode, oNode):
        """remove a given param from the network by masking its connection"""
        # obtain in-module and index
        iMod, iModName, iModNo, iNodeID = self._nameParse(iNode)
        oMod, oModName, oModNo, oNodeID = self._nameParse(oNode)
        # if a _F full connection
        if np.all(self.layerConns[iModNo]):
            assert len(self.connections[iMod]) == 1
            theConn = self.connections[iMod][0]
        # if a set of smaller full connections
        else:
            # obtain desired connection
            for conn in self.connections[iMod]:
                if conn.name == iNode + "_" + oNode:
                    theConn = conn
        # find parameter ID and mask
        paramID = _convNodeToParam(theConn, iNodeID, oNodeID)
        print(theConn.params)
        print(theConn.mask)
        print("given " + iNode + "_" + oNode + ", removing param " + \
                                        str(paramID) + " from " + theConn.name)
        theConn.maskParam(paramID)
        print(theConn.params)
        print(self.params)
        self.sortModules(forMask=True)
        print("masking network params")
        print(self.params)
        print(theConn.mask)
        print(theConn.params)
        return theConn

    def _nameParse(self, nName):
        # returns module, mod name, mod index, and node index, given node name
        for ix in range(len(nName)):
            if nName[ix] == "-":
                modName = nName[0:ix]
                nodeID = int(nName[ix+1:len(nName)])
        count = 0
        for mod in self.modulesSorted:
            if mod.name == modName:
                theMod = mod
                modNo = count
            count += 1
        return theMod, modName, modNo, nodeID

    def sortModules(self, forMask=False):
        """Prepare the network for activation by sorting the internal
        datastructure.
        Overrides automatic return if already sorted to allow for setting of
        masked params on NETWORK level"""
        if forMask==True:
            self.sorted = False
        pbFFNet.sortModules(self)

    def nxGraph(self):
        """omits edges whose weights are precisely zero"""
        #list of nodes
        nlist = [ [_nxNN(mod.name,d) for d in range(mod.dim)] \
                           for mod in self.modulesSorted]
        # list of edges w/ weights: (iNodeName, oNodeName, {'weight':wt}) tuple
        elist = [ [ [ (_nxNN(m1.name,_convParamToNode(c,p)[0]), \
                       _nxNN(m2.name,_convParamToNode(c,p)[1]), \
                       {'weight' : c.params[p]} ) \
                        for p in range(len(c.params)) if not c.params[p] == 0] \
                        for c in self.connections[m1] ] \
                        for m1, m2 in list(zip(self.modulesSorted[0:self.nHL+1], \
                                               self.modulesSorted[1:self.nHL+2])) ]
        # flatten edge and node lists
        fnlist = sum(nlist,[])
        felist = sum(sum(elist,[]),[])
        # add components to graph
        G = nx.DiGraph()
        G.add_nodes_from(fnlist)
        G.add_edges_from(felist)
        return G
