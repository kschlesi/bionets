# python functions for a hopfield network

import math
import numpy
import random

class HopfieldNet:
    def __init__(self,N,thresh,W):
        self.N = N
        self.state = numpy.zeros((N,1))
        if len(thresh)==N:
            self.thresh = thresh
        else:
            self.thresh = [0.0]*N
        if type(W)==numpy.ndarray and W.shape==(N,N):
                self.W = W
        else:
            self.W = numpy.zeros((N,N))

    def set_state(self,instate):
        if instate.shape==self.state.shape:
            self.state = instate

    def get_state(self):
        return self.state

    def print_state(self):
        print(self.state)

    def teach(self,patterns):
        if patterns.shape[0]==self.N:
            patterns = 2*patterns-1
            for i in range(self.N):
                for j in range(self.N):
                    if i==j:
                        self.W[i,j] = 0
                    else:
                        self.W[i,j] = numpy.sum(patterns[i,:]*patterns[j,:])

    def associate(self,pattern):
        # begin by setting original pattern
        self.set_state(pattern)
        a = range(self.N)
        # start loop of asynchronous updates
        cont = True;
        while cont:
            # randperm the nodes
            #order = random.sample(a,len(a))
            order = [1,3,2,4,0]
            # save old state
            old_state = self.state
            # increment each node in turn
            for node in order:
                self.increment(node)
            # check whether new state equals old state
            if numpy.all(self.state==old_state):
                cont = False
        self.print_state

    def increment(self,node):
        inval = numpy.dot(self.W[node,:],self.state)
        if inval[0]<self.thresh[node]:
            self.state[node] = 0
        else:
            self.state[node] = 1
