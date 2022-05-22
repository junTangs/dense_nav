
from rl.model.feature_model.feature_extractor import FeatureExtractor
from rl.common.deep_model import MLP
from torch import FloatTensor
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

class MLPFE(FeatureExtractor):
    def __init__(self,config):
        super(MLPFE, self).__init__(config)
        self.mlp = MLP(self.config["mlp_layer"])

    def feature(self,*args,**kwargs):
        x = args[0]
        x = x.view(x.shape[0],-1)
        return self.mlp(x)

    def preprocess(self,states):
        res = []
        for item in states:
            res.append([item["states"]])
        res = np.concatenate(res,axis=0)
        return Variable(FloatTensor(res)).to(self.device)

