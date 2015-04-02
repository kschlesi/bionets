# class Linear Masked Layer
# inherits from LinearLayer and MaskedModule

#from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from maskedparams_pc import MaskedParametersPC
from pybrain.structure import FullConnection
from numpy import zeros
import types

class FullMaskedConnection(FullConnection,MaskedParametersPC):

    ### ensure MaskedParameters.randomize() will initialize a mask with all bits on
    maskOnProbability = 0

    def __init__(self, *args, **kwargs):
        self.randomize = types.MethodType(FullConnection.randomize, self)
        FullConnection.__init__(self, *args, **kwargs)
        self.randomize = types.MethodType(MaskedParametersPC.randomize, self)
        MaskedParametersPC.__init__(self, self)
        self.randomize = types.MethodType(FullConnection.randomize, self)

    ### method to mask a particular parameter
    def maskParam(self,paramID):
        paramcount = 0
        for i in range(len(self.maskableParams)):
            if self.mask[i] == True:
                if paramcount == paramID:
                    self.mask[i] == False
                paramcount += 1
        self._applyMask()

    ### method to unmask a particular parameter
    # def unMaskParam(self,paramID):
    #     paramcount = 0
    #     for i in range(len(self.maskableParams)):
    #         if self.mask[i] == True:
    #             if any(paramcount == paramID):
    #                 self.mask[i] == False
    #             paramcount += 1
    #     self._applyMask()
