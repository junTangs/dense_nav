
from rl.model.feature_model.feature_extractor import FeatureExtractor
from rl.common.deep_model import BasicBlock1d,ChannelAttention,MLP,SelfAttention
from torch import FloatTensor
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np

class CA1DFusionSA(FeatureExtractor):
    def __init__(self,config):
        super(CA1DFusionSA, self).__init__(config)


        self.convs = nn.ModuleList()
        self.atts = nn.ModuleList()
        for i,conf in enumerate(self.config["conv_layers"]):
            att_conf = {"in_channel":conf["out_channel"],"r":2}
            self.convs.add_module(f"base_block_{i}",BasicBlock1d(conf["in_channel"],conf["out_channel"],downsample=conf["downsample"]))
            self.atts.add_module(f"atten_{i}",ChannelAttention(att_conf))
            

        self.sensor_sa_config = self.config["sensor_sa"]
        self.state_sa_config = self.config["state_sa"]

        self.state_encoder_config = self.config["state_layers"]
        self.state_encoder = MLP(self.state_encoder_config)
        self.sensor_sa = SelfAttention(self.sensor_sa_config["in_features"],self.sensor_sa_config["embed_features"],short_cut=True)
        self.state_sa = SelfAttention(self.state_sa_config["in_features"],self.state_sa_config["embed_features"],short_cut=True)

        self.fusion_mlp = MLP(self.config["fusion_layers"])

        self.sensor_dim = self.config["sensor_dim"]



    def feature(self,*args,**kwargs):

        x_s = args[0][0]
        # batch_size,seq_len*sensor_dim
        x_r = args[0][1]

        b,l,_ = x_r.shape

        # space feature
        frames = []
        for i in range(l):
            frame = x_s[:,i,:].unsqueeze(1)
            for conv,attn in zip(self.convs,self.atts):
                frame = conv(frame)
                frame = attn(frame)
                frames.append(frame.view(b,1,-1))

        x_s = torch.cat(frames,dim=1)

        x_r = self.state_encoder(x_r)
        # xs: [b,seq,dim] -> [b,seq,embed_feature]

        x_s = self.sensor_sa(x_s).view(b,-1)
        x_r = self.state_sa(x_r).view(b,-1)
        x = self.fusion_mlp(torch.cat([x_s,x_r],dim=1))
        return x

    def preprocess(self,states):
        obsv_res = []
        state_res = []

        for item in states:
            obsv_res.append([item["states"][:,-self.sensor_dim:]])
            state_res.append([item["states"][:,:-self.sensor_dim]])
        obsv_res = np.concatenate(obsv_res,axis=0)
        state_res = np.concatenate(state_res,axis=0)
        return Variable(FloatTensor(obsv_res)).to(self.device),Variable(FloatTensor(state_res)).to(self.device)

