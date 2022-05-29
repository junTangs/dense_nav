import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class SelfAttention(nn.Module):
    def __init__(self,in_feature,embed_feature,short_cut = False):
        super(SelfAttention, self).__init__()

        self.q_linear = nn.Linear(in_feature,embed_feature)
        self.k_linear = nn.Linear(in_feature,embed_feature)
        self.v_linear = nn.Linear(in_feature,embed_feature)
        if in_feature != embed_feature and short_cut is True:
            self.short_cut_linear = nn.Linear(in_feature,embed_feature)
        else:
            self.short_cut_linear = None
        self.softmax= nn.Softmax(dim=-1)
        self.short_cut = False

    def forward(self,x):
        # x: [batch,seq,dim]
        q  = self.q_linear(x)
        k  = self.k_linear(x)
        v = self.v_linear(x)

        b,seq,dim_k = k.shape
        # q,k,v : [batch,seq,em_dim]
        # dot : [batch,seq,seq]
        dot = torch.bmm(q,torch.permute(k,(0,2,1)))
        dot = dot/math.sqrt(dim_k)
        att = self.softmax(dot)
        out = torch.bmm(att,v)
        if self.short_cut:
            if self.short_cut_linear is not None:
                out += self.short_cut_linear(x)
            else:
                out += x
        return out











if __name__ == "__main__":
    sa = SelfAttention(3,2)
    x = torch.randn(2,3,3)
    sa(x)
