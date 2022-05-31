import torch
import torch.nn as nn


class SpacialAttention2d(nn.Module):
    def __init__(self, kernel=3):
        super(SpacialAttention2d, self).__init__()

        assert kernel in [3, 7], "[SpacialAttention]: kernal must 3 or 7"
        self.padding = 3 if kernel == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size=kernel,padding=self.padding,bias=False)
        self.relu = nn.ReLU()

    def forward(self,x):
        mean = torch.mean(x,1,keepdim=True)
        max,_ = torch.max(x,1,keepdim=True)
        out = self.relu(self.conv(torch.cat([mean,max],dim=1)))
        return out

if __name__ == "__main__":
    x = torch.randn((32,3,10,10))
    sa = SpacialAttention2d(3)
    x = sa(x)
    print(x.shape)
