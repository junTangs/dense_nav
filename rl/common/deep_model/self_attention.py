
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from rl.common.deep_model import MLP



class MultiSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        # multi head num
        self.num_multi_heads = config["num_multi_heads"]
        # input feature dims
        self.dims = config["dims"]
        # hidden feature dims
        self.hidden_dims = config["hidden_dims"]
        # hidden feature dims of each head
        self.attn_hidden_dims = int(self.hidden_dims/self.num_multi_heads)


        assert(self.hidden_dims % self.num_multi_heads == 0 ),"[SelfAttention]: hidden dims must be divisible by head num"


        # net structure
        # embedding layers
        self.q_layer = nn.Linear(self.dims,self.hidden_dims)
        self.k_layer = nn.Linear(self.dims,self.hidden_dims)
        self.v_layer = nn.Linear(self.dims,self.hidden_dims)
        self.out_layer = nn.Linear(self.hidden_dims,self.dims)
        self.layer_norm = nn.LayerNorm(self.dims)

    def scale_dot_attention(self,q,k,v):
        # Q,K,V : (b,seq_len,attn_hidden_dims)
        q_k_mul = torch.bmm(q,k.permute(0,2,1))
        attn = torch.softmax(q_k_mul,dim=-1)/math.sqrt(k.shape[-1])
        context = torch.bmm(attn,v)
        return context,attn

    def forward(self,q,k,v):
        batch_size = k.shape[0]
        seq_length = k.shape[1]
        # embeding : (b,l,d) -> (b,l,hidden_dims)
        q = self.q_layer(q)
        k = self.k_layer(k)
        v = self.v_layer(v)

        q = q.view(batch_size,-1,seq_length,self.attn_hidden_dims)
        k = k.view(batch_size,-1,seq_length,self.attn_hidden_dims)
        v = v.view(batch_size,-1,seq_length,self.attn_hidden_dims)

        concat_context = []
        attns = []
        for i in range(self.num_multi_heads):
            q_i = q[:,i,:,:]
            k_i = k[:,i,:,:]
            v_i = v[:,i,:,:]
            context,attn = self.scale_dot_attention(q_i,k_i,v_i)
            concat_context.append(context)
            attns.append(attns)

        concat_context = torch.cat(concat_context,-1)

        return self.out_layer(concat_context),attns


class MSA(MultiSelfAttention):
    def __init__(self,config):
        super().__init__(config)

    def forward(self,x):
        x,_ =  super().forward(x,x,x)
        return x

class PositionEncoding(nn.Module):
    def __init__(self,config):
        super(PositionEncoding,self).__init__()
        self.config = config
        self.dropout = nn.Dropout(self.config["dropout"])
        self.P = torch.zeros((1,self.config["max_len"],self.config["hidden_dims"]))

        X = torch.arange(self.config["max_len"],dtype=torch.float32).reshape(-1,1)/torch.pow(10000,torch.arange(0,self.config["hidden_dims"],2,dtype=torch.float32)/self.config["hidden_dims"])
        self.P[:,:,0::2] = torch.sin(X)
        self.P[:,:,1::2] = torch.cos(X)

    def forward(self,x):
        x = x + self.P[:,:x.shape[1],:].to(x.device)
        return self.dropout(x)

if __name__ == "__main__":

    config = {"num_multi_heads":8,"dims":8,"hidden_dims":32}
    q = torch.ones((32,3,8))
    k = torch.ones((32, 3, 8))
    v = k
    pe_config = {"max_len": 1000, "dropout": 0.3, "hidden_dims": 8}
    pe = PositionEncoding(pe_config)
    self_attention = MultiSelfAttention(config)

    x,_= self_attention(q,k,v)
    print(x.shape)
    x = pe(x)
    print(x.shape)



