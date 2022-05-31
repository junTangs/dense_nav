
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from .mlp import MLP


class ChannelAttention(nn.Module):
    def __init__(self, config):
        super(ChannelAttention, self).__init__()
        self.config = config
        self.in_channel = config["in_channel"]
        self.r = config["r"]

        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.mlp_config = {
            "in_channel": self.in_channel,
            "out_channel": self.in_channel,
            "layers": [self.in_channel // self.r, self.in_channel],
            "activation": "relu",
            "out_activation": "sigmoid"
        }

        self.mlp = MLP(self.mlp_config)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:[b,c,l]
        average_pool = self.average_pool(x)
        max_pool = self.max_pool(x)

        # [b,c]
        average_pool = self.mlp(average_pool.squeeze(-1))
        max_pool = self.mlp(max_pool.squeeze(-1))

        # [b,c]
        out = average_pool + max_pool
        out = out.unsqueeze(-1)

        return self.sigmoid(out) * x

class ChannelAttention2d(nn.Module):
    def __init__(self, config):
        super(ChannelAttention2d, self).__init__()
        self.config = config
        self.in_channel = config["in_channel"]
        self.r = config["r"]

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp_config = {
            "in_channel": self.in_channel,
            "out_channel": self.in_channel,
            "layers": [self.in_channel // self.r, self.in_channel],
            "activation": "relu",
            "out_activation": "sigmoid"
        }

        self.mlp = MLP(self.mlp_config)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:[b,c,h,w]
        b,c,h,w = x.shape
        average_pool = self.average_pool(x)
        max_pool = self.max_pool(x)
        # [b,c]
        average_pool = self.mlp(average_pool.view(b,-1))
        max_pool = self.mlp(max_pool.view(b,-1))

        # [b,c]
        out = average_pool + max_pool
        out = out.view(b,c,1,1)
        return self.sigmoid(out) * x