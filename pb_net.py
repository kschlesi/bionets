### BIONETS NEURAL NETWORKS using PYBRAIN

# This code defines a class of FF/BP neural network for the bionets project.
# Network properties include an arbitrary number of hidden layers and nodes
# per layer, an input and output layer with bias nodes, the
# ability to individually define layer-to-layer connections.
# The network is trained with:
# The network can be dynamically attacked or degraded by removing nodes in
# between individual training sessions.

import math
import numpy as np
import random
#import matplotlib as mpl
import matplotlib.pyplot as pp
import scipy as sp
import pybrain as pb
import networkx as nx

class pbFFNet:

    def __init__(self, layers, nName="net", layerTypes=["lin","sig","sig"]):
        # layers = list with number of nodes per layer
        # no. layers= len(layers), no. nodes= sum(layers)
        # layers(1) = input layer, layers(end) = output layer
        self.layers = layers
        self.nHL = len(layers)-2
        self.nNodes = sum(layers)

        # initialize and name all nodes (default: input linear, others sigmoid)
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
            return fcxn
        else:
            cxns = []
            for i in range(np.shape(m)[0]):
                for j in range(np.shape(m)[1]):
                    if m[i,j]==1:
                        cName = L1 + "-" + str(i) + "_" + L2 + "-" + str(j)
                        cxn = pb.structure.FullConnection(self.net[L1], self.net[L2], \
                                                      name=cName, \
                                                      inSliceFrom=i, inSliceTo=i+1, \
                                                      outSliceFrom=j, outSliceTo=j+1)
                        self.net.addConnection(cxn)
                        cxns = cxns + [cxn]
            return cxns

    def dispLayers(self):
        self.net.sortModules()
        return self.net.modulesSorted

    def dispNet(self):
        for mod in self.net.modulesSorted:
            print(mod)
            for conn in self.net.connections[mod]:
                for p in range(len(conn.params)):
                    iNode, oNode = _convParamToNode(conn,p)
                    print("(" + str(iNode) + ", " \
                              + str(oNode) + ")", \
                                conn.params[p])

    def dispWeights(self):
        for mod in self.net.modules:
            for conn in self.net.connections[mod]:
                #print(conn)
                for cc in range(len(conn.params)):
                    print(conn.whichBuffers(cc), conn.params[cc])

    #def removeNode(self,lName,node):
        # remove a given node from the network while keeping existing params


    def nxGraph(self):
        #list of nodes
        nlist = [ [_nxNN(mod.name,d) for d in range(mod.dim)] \
                           for mod in self.net.modulesSorted]
        # list of edges with weights: (iNodeName, oNodeName, weight) tuple
        elist = [ [ [ (_nxNN(m1.name,_convParamToNode(c,p)[0]), \
                       _nxNN(m2.name,_convParamToNode(c,p)[1]), \
                       {'weight' : c.params[p]} ) \
                        for p in range(len(c.params)) ] \
                        for c in self.net.connections[m1] ] \
                        for m1, m2 in list(zip(self.net.modulesSorted[0:self.nHL+1], \
                                               self.net.modulesSorted[1:self.nHL+2])) \

                ]
        # flatten edge and node lists
        fnlist = sum(nlist,[])
        felist = sum(sum(elist,[]),[])
        # add components to graph
        G = nx.DiGraph()
        G.add_nodes_from(fnlist)
        G.add_edges_from(felist)
        return G

    def netPlot(self,layout=None):
        G = self.nxGraph()
        if layout is None:
            nx.draw(G) # and more things
        else:
            pos = self._nodePositions(layout)
            nx.draw(G,pos)
        pp.show()

    def _nodePositions(self,layout):
        # takes an nx graph with the _nxNN node naming conventions and a layout
        # keystring; returns position variable to use with nx.draw
        if layout == "Layers":
            pos = [ [ [_nxNN(self.net.modulesSorted[lx].name,d), (lx,d)] \
                        for d in range(self.net.modulesSorted[lx].dim) ] \
                        for lx in range(len(self.layers)) ]
            pos = sum(pos,[])
            pdic = { name : coor for name, coor in pos}
        return pdic

################################################################################

def _convParamToNode(conn,p):
    # works for any FullConnection object and a p index of param #
    iT = conn.inSliceTo
    iF = conn.inSliceFrom
    oT = conn.outSliceTo
    oF = conn.outSliceFrom
    iNode = (p % (iT-iF))+iF
    oNode = math.floor((p/(iT-iF)))+oF
    return iNode, oNode


def _nxNN(mname,nID):
    return mname+"-"+str(nID)
