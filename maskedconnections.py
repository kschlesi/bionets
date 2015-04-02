# class Linear Masked Layer
# inherits from LinearLayer and MaskedModule

#from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from maskedparams_pc import MaskedParametersPC
from pybrain.structure import FullConnection
from pybrain.structure.parametercontainer import ParameterContainer
from numpy import zeros
import types

class FullMaskedConnection(FullConnection,MaskedParametersPC):

    ### ensure MaskedParameters.randomize() will initialize a mask with all bits on
    maskOnProbability = 0
    returnZeros = True  # to prevent recursion loop

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
                    self.mask[i] = False
                paramcount += 1
        self._applyMask()

    @property
    def params(self):
        """ returns an array with (usually) only the unmasked parameters """
        if self.returnZeros:
            return self._params
        else:
            x = zeros(self.paramdim)
            paramcount = 0
            for i in range(len(self.maskableParams)):
                if self.mask[i] == True:
                    x[paramcount] = self.maskableParams[i]
                    paramcount += 1
            return x


    ### method to unmask a particular parameter
    # def unMaskParam(self,paramID):
    #     paramcount = 0
    #     for i in range(len(self.maskableParams)):
    #         if self.mask[i] == True:
    #             if any(paramcount == paramID):
    #                 self.mask[i] == False
    #             paramcount += 1
    #     self._applyMask()
