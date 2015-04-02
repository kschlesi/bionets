### class that inherits from MaskedParameters and overrides functions
# necessary for functioning as a co-deriver from ParameterContainer AND
# MaskedParameters

from pybrain.structure.evolvables.maskedparameters import MaskedParameters

class MaskedParametersPC(MaskedParameters):

    def __init__(self, *args, **kwargs):
        MaskedParameters.__init__(self, *args, **kwargs)

    def _setParameters(self, x, owner=None):
        MaskedParameters._setParameters(self, x)
