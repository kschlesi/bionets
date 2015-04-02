# class Linear Masked Layer
# inherits from LinearLayer and MaskedModule

#from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from pybrain.structure.evolvables.maskedmodule import MaskedModule
from pybrain.structure import LinearLayer, SigmoidLayer
from numpy import zeros

class FullMaskedConnection(FullConnection,MaskedParameters):

    ### ensure self.randomize() will initialize a mask with all bits on
    maskOnProbability = 1

    def __init__(self, *args, **kwargs):
        self.mask = zeros(dim, dtype=bool)
        FullConnection.__init__(self, *args, **kwargs)
        MaskedModule.__init__(self, self)

    ### method to mask a particular parameter
    def maskParam(self,paramID):
        paramcount = 0
        for i in range(len(self.maskableParams)):
            if self.mask[i] == True:
                if any(paramcount == paramID):
                    self.mask[i] == False
                paramcount += 1
        self._applyMask()

    ### re-override _resetBuffers to Module version
    def _resetBuffers(self, length=1):
        """Reset buffers to a length (in time dimension) of 1."""
        for buffername, dim in self.bufferlist:
            setattr(self, buffername, zeros((length, dim)))
        if length==1:
            self.offset = 0



class SigmoidMaskedLayer(SigmoidLayer,MaskedModule):

    ### ensure self.randomize() will initialize a mask with all bits on
    maskOnProbability = 1

    def __init__(self, dim, *args, **kwargs):
        self.mask = zeros(dim, dtype=bool)
        SigmoidLayer.__init__(self, dim, *args, **kwargs)
        MaskedModule.__init__(self, self)

    ### method to mask a particular parameter
    def maskParam(self,paramID):
        paramcount = 0
        for i in range(len(self.maskableParams)):
            if self.mask[i] == True:
                if any(paramcount == paramID):
                    self.mask[i] == False
                paramcount += 1
        self._applyMask()

    ### re-override _resetBuffers to Module version
    def _resetBuffers(self, length=1):
        """Reset buffers to a length (in time dimension) of 1."""
        for buffername, dim in self.bufferlist:
            setattr(self, buffername, zeros((length, dim)))
        if length==1:
            self.offset = 0
