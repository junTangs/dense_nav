import torch.nn as nn
import torch.nn.functional as F
import math
import torch


class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,downsample = 1):
        super(BasicBlock, self).__init__()

        self.conv3x3 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=downsample),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, 3,  padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=1),
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
                nn.ReLU(),
                nn.Conv1d(in_channel, out_channel, 3, padding=1, stride=downsample),
                nn.ReLU(),
                nn.Conv1d(out_channel, out_channel, 3,  padding=1, stride=1),
                nn.ReLU(),
                nn.Conv1d(out_channel, out_channel, 3, padding=1, stride=1),
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
    net = BasicBlock1d(3,3,2)
    x = torch.randn(32,3,7)
    x = net(x)
    print(x.shape)






