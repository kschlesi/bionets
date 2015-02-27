### BIONETS NEURAL NETWORKS

# This code defines a class of FF/BP neural network for the bionets project.
# Network properties include an arbitrary number of hidden layers with equal
# number of nodes in each, an input and output layer with bias nodes, the
# ability to individually define layer-to-layer connections.
# The network is trained with conjugate gradient descent using Polak-Ribiere
# updating.
# The network can be dynamically attacked or degraded by removing nodes in
# between individual training sessions.

import math
import numpy as np
import random
import matplotlib as mpl

# sigmoid activation function
def sig(x):
    return 1/(1+np.exp(-x))

# derivative of activ. in terms of output (y)
def dsig_y(y):
    return y*(1-y)

class BioNet:
    def __init__(self, nI, nO, hidden, cAll='All', rr=1.0):
        # number of input nodes, output nodes, and list with number of nodes
        # in each hidden layer. no. layers= len(hidden), no. nodes= sum(hidden)
        self.nI = nI
        self.nO = nO
        self.hidden = hidden
        self.nHL = len(hidden) # no. hidden layers
        self.nHN = sum(hidden) # total no. hidden nodes
        self.mHL = max(hidden) # max no. hidden nodes per layer

        # must store: (1) the inherent state of each node (a)
        #             (2) the connectivity (c) and weights (w) between nodes

        # ACTIVATION: initialize all nodes to 1.0 activation
        self.aI = np.ones[nI+1,1] # +1 for bias node
        self.aO = np.ones([nO,1])
        self.aH = np.zeros([self.mHL+1,self.nHL])
        for iL in range(self.nHL):
            self.aH[0:hidden[iL]+1,iL] = 1.0

        # CONNECTIVITY: initialize to cAll (+1 for bias node in each layer)
        self.cI = np.ones([nI+1,hidden[0]]) # full connectivity for input layer
        self.cO = np.ones([hidden[self.nHL-1]+1,nO]) # and for output layer
        self.cH = np.zeros([self.mHL+1,self.mHL,self.nHL-1]) # zeros for others
        # for connections between each pair of hidden layers:
        if cAll=='All':
            cAll = np.ones([self.mHL,self.mHL,self.nHL-1]) # default all-to-all
        for iL in range(self.nHL-1):
            # set all extant nodes but bias node to input
            self.cH[1:hidden[iL]+1,0:hidden[iL+1],iL] = cAll[0:hidden[iL],0:hidden[iL+1],iL]
            # set bias nodes to all-connect
            self.cH[0,:,iL] = 1

        # WEIGHTS: initialize each extant connection to a random value in [-1,1]
        self.wH = randNonZeroFill(self.cH,-rr,rr)
        self.wI = randNonZeroFill(self.cI,-rr,rr)
        self.wO = randNonZeroFill(self.cO,-rr,rr)

    def removeNode(self,nodelayer,nodeID):
        # enter node layer (number hidden layers starting at 0) and
        # node ID within hidden layer (number bias node as 0)
        self.hidden[nodelayer] = self.hidden[nodelayer]-1  # update node numbers
        self.aH[nodeID,nodelayer] = 0  # set activation to 0
        # update connectivities
        if nodelayer==0:
            self.cH[nodeID,:,nodelayer] = 0
            if nodeID!=0:
                self.cI[:,nodeID-1] = 0
        elif nodelayer==self.nHL-1:
            self.cO[nodeID,:] = 0
            if nodeID!=0:
                self.cH[:,nodeID-1,nodelayer-1] = 0
        else:
            self.cH[nodeID,:,nodelayer] = 0
            if nodeID!=0:
                self.cH[:,nodeID-1,nodelayer-1] = 0
        # zero erased weights
        self.wH[self.cH==0] = 0;

    def feedForward(self,x):
        # x is a training example (ndarray); len(x) should = nI
        if len(x) != self.ni:
            raise ValueError('wrong number of inputs')
        # set input node activations
        aI[1:nI+1] = x
        # propagate through hidden layers
        for iL in range(nHL):



    def dispNet(self):
        # plots a visualization of the network (with matplotlib?)
        print(self.aI)
        print(self.aH)
        print(self.aO)
        # create scatter vectors
        #vec1 = np.where(aH || aH==0)




def randNonZeroFill(inmat,rlo,rhi):
# creates ndarray of same size as ndarray 'inmat', and fills all entries where
# inmat!=0 with uniform random number in range [-rg,rg)
    outmat = np.empty_like(inmat)
    outmat[inmat==1] = np.random.uniform(rlo,rhi,len(inmat[inmat==1]))
    return outmat
