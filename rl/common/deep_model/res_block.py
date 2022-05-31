import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from rl.common.deep_model.spacial_attention import SpacialAttention2d
from rl.common.deep_model.channel_attention import ChannelAttention2d

class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,downsample = 1,kernel = 3):
        super(BasicBlock, self).__init__()

        padding = 1 if kernel == 3 else 3
        self.conv3x3 = nn.Sequential(
                # nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, out_channel, kernel, padding=padding, stride=downsample),
                # nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, kernel,  padding=padding, stride=1),
        )

        if in_channel != out_channel or downsample!=1:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,1,padding=0,stride=downsample),
                nn.ReLU()
            )
        else:
            self.conv1x1 = None

        return


    def forward(self,x):
        out = self.conv3x3(x)
        shortcut = x
        if self.conv1x1 is not None:
            shortcut = self.conv1x1(shortcut)
        return out+shortcut




class BasicBlockV2(nn.Module):
    def __init__(self,in_channel,out_channel,downsample = 1,kernel = 3):
        super(BasicBlockV2, self).__init__()

        padding = 1 if kernel == 3 else 3
        self.conv3x3 = nn.Sequential(
                # nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, out_channel, kernel, padding=padding, stride=downsample),
                # nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, kernel,  padding=padding, stride=1),
                ChannelAttention2d({"in_channel":out_channel,"r":2}),
                SpacialAttention2d()
        )

        if in_channel != out_channel or downsample!=1:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,1,padding=0,stride=downsample),
                nn.ReLU()
            )
        else:
            self.conv1x1 = None

        return


    def forward(self,x):
        out = self.conv3x3(x)
        shortcut = x
        if self.conv1x1 is not None:
            shortcut = self.conv1x1(shortcut)
        return out+shortcut


class BasicBlock1d(nn.Module):
    def __init__(self,in_channel,out_channel,downsample = 1):
        super(BasicBlock1d, self).__init__()

        self.conv3x3 = nn.Sequential(
                # nn.BatchNorm1d(in_channel),
                nn.ReLU(),
                nn.Conv1d(in_channel, out_channel, 3, padding=1, stride=downsample),
                # nn.BatchNorm1d(out_channel),
                nn.ReLU(),
                nn.Conv1d(out_channel, out_channel, 3,  padding=1, stride=1),
        )

        if in_channel != out_channel or downsample!=1:
            self.conv1x1 = nn.Sequential(
                nn.Conv1d(in_channel,out_channel,1,padding=0,stride=downsample),
                nn.ReLU()
            )
        else:
            self.conv1x1 = None

        return


    def forward(self,x):
        out = self.conv3x3(x)
        shortcut = x
        if self.conv1x1 is not None:
            shortcut = self.conv1x1(shortcut)
        return out+shortcut


if __name__ == "__main__":
    net = BasicBlockV2(3,3,4)
    x = torch.randn(32,3,16,16)
    x = net(x)
    print(x.shape)






