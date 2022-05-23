from rl.model.feature_model.ca1d import CA1D
from rl.model.feature_model.ccsa1d import CCSA1D
from rl.model.feature_model.mlpfe import MLPFE
from rl.model.feature_model.ca1d_fusion import CA1DFusion


FEATURE_EXTRACTOR_FACTORY = {"ca1d":CA1D,"ccsa1d":CCSA1D,"mlpfe":MLPFE,"ca1df":CA1DFusion}