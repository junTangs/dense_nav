
from rl.model.feature_model.feature_extractor import FeatureExtractor
from rl.common.deep_model import BasicBlock1d,ChannelAttention,MLP
from torch import FloatTensor
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np

class CA1DFusion(FeatureExtractor):
    def __init__(self,config):
        super(CA1DFusion, self).__init__(config)


        self.convs = nn.ModuleList()
        self.atts = nn.ModuleList()
        for i,conf in enumerate(self.config["conv_layers"]):
            att_conf = {"in_channel":conf["out_channel"],"r":2}
            self.convs.add_module(f"base_block_{i}",BasicBlock1d(conf["in_channel"],conf["out_channel"],downsample=conf["downsample"]))
            self.atts.add_module(f"atten_{i}",ChannelAttention(att_conf))
            
            
        self.s_r_mlp = MLP(self.config["mlp_layers"])
        self.fusion_mlp = MLP(self.config["fusion_mlp_layers"])
            
        self.sensor_dim = self.config["sensor_dim"]

    def feature(self,*args,**kwargs):
        x_s = args[0]
        # batch_size,seq_len*sensor_dim
        x_r = args[1].view(x_r.shape[0],-1)
        for conv,attn in zip(self.convs,self.atts):
            x_s = conv(x_s)
            x_s = attn(x_s)
            
        x_r = self.s_r_mlp(x_r)
        x = self.fusion_mlp(torch.cat([x_s,x_r],dim=1))
        return x

    def preprocess(self,states):
        obsv_res = []
        state_res = []
        for item in states:
            obsv_res.append([item["states"][:,:,:self.sensor_dim]])
            state_res.append([item["states"][:,:,self.sensor_dim:]])
        obsv_res = np.concatenate(obsv_res,axis=0)
        state_res = np.concatenate(state_res,axis=0)
        return Variable(FloatTensor(obsv_res)).to(self.device),Variable(FloatTensor(state_res)).to(self.device)

