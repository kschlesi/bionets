# class Linear Masked Layer
# inherits from LinearLayer and MaskedModule

from pybrain.structure.evolvables.maskedmodule import MaskedModule
from pybrain.structure import LinearLayer

class LinearMaskedLayer(MaskedModule,LinearLayer):

    ### ensure self.randomize() will initialize a mask with all bits on
    self.maskOnProbability = 1

    ### method to mask a particular parameter
    def maskParam(self,paramID):
        paramcount = 0
        for i in range(len(self.maskableParams)):
            if self.mask[i] == True:
                self.maskableParams[i] = x[paramcount]n.mask
                paramcount += 1
        self._applyMask()
