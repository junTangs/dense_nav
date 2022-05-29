
from rl.model.feature_model.feature_extractor import FeatureExtractor
from rl.common.deep_model import BasicBlock1d,ChannelAttention,MLP
from torch import FloatTensor
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np

class CA1DFusionLSTM(FeatureExtractor):
    def __init__(self,config):
        super(CA1DFusionLSTM, self).__init__(config)


        self.convs = nn.ModuleList()
        self.atts = nn.ModuleList()
        for i,conf in enumerate(self.config["conv_layers"]):
            att_conf = {"in_channel":conf["out_channel"],"r":2}
            self.convs.add_module(f"base_block_{i}",BasicBlock1d(conf["in_channel"],conf["out_channel"],downsample=conf["downsample"]))
            self.atts.add_module(f"atten_{i}",ChannelAttention(att_conf))

        self.fusion_mlp = MLP(self.config["fusion_layers"])

        self.sensor_lstm_config = self.config["sensor_lstm"]
        self.lstm1 = nn.GRU(input_size=self.sensor_lstm_config["input_size"],
                            hidden_size=self.sensor_lstm_config["hidden_size"],
                            num_layers= self.sensor_lstm_config["num_layers"],
                            batch_first=True)

        self.state_lstm_config = self.config["state_lstm"]
        self.lstm2 = nn.GRU(input_size=self.state_lstm_config["input_size"],
                            hidden_size=self.state_lstm_config["hidden_size"],
                            num_layers= self.state_lstm_config["num_layers"],
                            batch_first=True)

        self.sensor_dim = self.config["sensor_dim"]



    def feature(self,*args,**kwargs):

        x_s = args[0][0]
        # batch_size,seq_len*sensor_dim
        x_r = args[0][1]

        b,l,_ = x_r.shape

        h1  = torch.zeros((self.sensor_lstm_config["num_layers"], b, self.sensor_lstm_config["hidden_size"]), device=self.device)
        h2  = torch.zeros((self.state_lstm_config["num_layers"], b, self.state_lstm_config["hidden_size"]), device=self.device)

        # space feature
        frames = []
        for i in range(l):
            frame = x_s[:,i,:].unsqueeze(1)
            for conv,attn in zip(self.convs,self.atts):
                frame = conv(frame)
                frame = attn(frame)
                frames.append(frame.view(b,1,-1))
        x_s = torch.cat(frames,dim=1)


        # xs: b channel dim -> n b hidden_size
        _,h1_t = self.lstm1(x_s,h1)
        _,h2_t = self.lstm2(x_r,h2)

        # b * hidden_size
        x_s = h1_t[-1,:,:]
        x_r = h2_t[-1,:,:]
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

