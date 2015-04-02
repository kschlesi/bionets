# class Linear Masked Layer
# inherits from LinearLayer and MaskedModule

from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from pybrain.structure import LinearLayer, SigmoidLayer

class LinearMaskedLayer(LinearLayer,MaskedParameters):

    ### ensure self.randomize() will initialize a mask with all bits on
    maskOnProbability = 1

    def __init__(self, dim):
        LinearLayer.__init__(self, dim, **args)
        MaskedParameters.__init__(self, self, **args)

    ### method to mask a particular parameter
    def maskParam(self,paramID):
        paramcount = 0
        for i in range(len(self.maskableParams)):
            if self.mask[i] == True:
                if paramcount == paramID:
                    self.mask[i] == False
                paramcount += 1
        self._applyMask()


class SigmoidMaskedLayer(SigmoidLayer,MaskedParameters):

    ### ensure self.randomize() will initialize a mask with all bits on
    maskOnProbability = 1

    def __init__(self, dim):
        SigmoidLayer.__init__(self, dim, **args)
        MaskedParameters.__init__(self, self, **args)

    ### method to mask a particular parameter
    def maskParam(self,paramID):
        paramcount = 0
        for i in range(len(self.maskableParams)):
            if self.mask[i] == True:
                if paramcount == paramID:
                    self.mask[i] == False
                paramcount += 1
        self._applyMask()
