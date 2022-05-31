
from rl.model.feature_model.feature_extractor import FeatureExtractor
from rl.common.deep_model import BasicBlock1d,ChannelAttention,SpacialAttention2d
from torch import FloatTensor
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

class CCSA1D(FeatureExtractor):
    def __init__(self,config):
        super(CCSA1D, self).__init__(config)


        self.convs = nn.ModuleList()
        self.atts = nn.ModuleList()
        self.sa_atts = nn.ModuleList()
        for i,conf in enumerate(self.config["conv_layers"]):
            att_conf = {"in_channel":conf["out_channel"],"r":2}
            self.convs.add_module(f"base_block_{i}",BasicBlock1d(conf["in_channel"],conf["out_channel"],downsample=conf["downsample"]))
            self.atts.add_module(f"atten_{i}",ChannelAttention(att_conf))
            self.sa_atts.add_module(f"sa_atten_{i}",SpacialAttention2d(3))

    def feature(self,*args,**kwargs):
        x = args[0]
        for conv,attn,sp_attn in zip(self.convs,self.atts,self.sa_atts):
            x = conv(x)
            x = attn(x)
            x = x*sp_attn(x)
        return x

    def preprocess(self,states):
        res = []
        for item in states:
            res.append([item["states"]])
        res = np.concatenate(res,axis=0)
        return Variable(FloatTensor(res)).to(self.device)

