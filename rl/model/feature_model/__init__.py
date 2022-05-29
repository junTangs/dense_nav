from rl.model.feature_model.ca1d import CA1D
from rl.model.feature_model.ccsa1d import CCSA1D
from rl.model.feature_model.mlpfe import MLPFE
from rl.model.feature_model.ca1d_fusion import CA1DFusion
from rl.model.feature_model.ca1d_lstm_fusion import CA1DFusionLSTM
from rl.model.feature_model.ca1d_sa_fusion import CA1DFusionSA


FEATURE_EXTRACTOR_FACTORY = {"ca1d":CA1D,"ccsa1d":CCSA1D,"mlpfe":MLPFE,"ca1df":CA1DFusion,"ca1df_lstm":CA1DFusionLSTM,
                             "ca1df_sa":CA1DFusionSA}