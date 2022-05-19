import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from rl.common.deep_model import NoisyFactorizedLinear

class NoisyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = config["layers"]
        self.in_channel = config["in_channel"]
        self.out_channel = config["out_channel"]
        self.activation = config["activation"]
        self.out_activation = config["out_activation"]

        self.mlp = nn.Sequential()

        last_channel = self.in_channel
        for i, channel in enumerate(self.layers):
            self.mlp.add_module("layer{}".format(i), NoisyFactorizedLinear(last_channel, channel))
            if self.activation == "relu":
                self.mlp.add_module("relu{}".format(i), nn.ReLU())
            elif self.activation == "tanh":
                self.mlp.add_module("tanh{}".format(i), nn.Tanh())
            elif self.activation == "sigmoid":
                self.mlp.add_module("sigmoid{}".format(i), nn.Sigmoid())
            else:
                self.mlp.add_module("relu{}".format(i), nn.ReLU())
            last_channel = channel

        self.mlp.add_module("layer_out", NoisyFactorizedLinear(last_channel, self.out_channel))

        if self.out_activation == "relu":
            self.mlp.add_module("relu_out", nn.ReLU())
        elif self.out_activation == "tanh":
            self.mlp.add_module("tanh_out", nn.Tanh())
        elif self.out_activation == "sigmoid":
            self.mlp.add_module("sigmoid_out", nn.Sigmoid())
        else:
            pass

    def forward(self, x):
        # x: [batch_size,in_channel]
        return self.mlp(x)
