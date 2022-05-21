from re import S
from rl.common.deep_model import MLP
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class ICM(nn.Module):
    def __init__(self,feature_extractor,feature_dims,action_dims,beta = 0.5,alpha = 1):
        super(ICM, self).__init__()
        self.feature_extractor = feature_extractor
        self.forward_model  = ForwardModel(feature_dims,action_dims)
        self.inverse_model = InvModel(feature_dims,action_dims)
        self.device = None

        self.action_dims = action_dims
        self.feature_dims = feature_dims
        self.beta = beta
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def set_device(self,device):
        self.device = device
        self.to(device)
        self.forward_model.set_device(device)
        self.inverse_model.set_device(device)
        return

    def get_feature(self,states):
        return self.feature_extractor(states)

    def pred_s_(self,s,a):
        return self.forward_model(s,a)


    def pred_a(self,s,s_):
        return self.inverse_model(s,s_)

    def forward(self,states,next_states,a):
        a = torch.LongTensor(a)
        a_one_hot = F.one_hot(Variable(a).to(self.device),self.action_dims)
        feature_s = self.get_feature(states)
        feature_s_ = self.get_feature(next_states)
        pred_s_ = self.pred_s_(feature_s,a_one_hot)
        pred_a = self.pred_a(feature_s_,feature_s)

        forward_loss = self.alpha*self.mse_loss(feature_s_,pred_s_)
        inverse_loss = self.cross_entropy(pred_a,a)

        intrinsic_r  = torch.mean(torch.abs(feature_s_ - pred_s_)**2)

        intrinsic_r = intrinsic_r -self.intrinsic_reward.min()/(self.intrinsic_reward.max() - self.intrinsic_reward.min()+1e-8)
        loss = (1-self.beta)*inverse_loss+self.beta*forward_loss
        return intrinsic_r ,loss, forward_loss,inverse_loss

class InvModel(nn.Module):
    def __init__(self,feature_dims,action_dims) -> None:
        super(InvModel,self).__init__()
        self.feature_dims = feature_dims
        self.action_dims = action_dims
        
        
        config = {"in_channel":feature_dims*2,
                  "out_channel":action_dims,
                  "layers":[feature_dims,feature_dims],
                  "activation":"relu",
                  "out_activation":None}
                  
        self.mlp = MLP(config)
        
    def forward(self,s,s_):
        # s: [batch_size,state_dim]
        # s_: [batch_size,state_dim]
        return self.mlp(torch.cat((s,s_),dim=1))
    
    
class ForwardModel(nn.Module):
    def __init__(self,feature_dims,action_dims) -> None:
        super(ForwardModel,self).__init__()
        self.feature_dims = feature_dims
        self.action_dims = action_dims
        
        in_dims = feature_dims+action_dims
        config = {"in_channel":in_dims,
                  "out_channel":feature_dims,
                  "layers":[in_dims],
                  "activation":"relu",
                  "out_activation":"relu"}
        
        self.mlp = MLP(config)
        self.out = nn.Linear(feature_dims+action_dims,feature_dims)
    
    def forward(self,s,a):
        # s: [batch_size,state_dim]
        # a: [batch_size,action_dim]
        s_ = self.mlp(torch.cat((s,a),dim=1))
        return self.out(torch.cat((s_,a),dim=1))
             
if __name__ == "__main__":
    a = torch.randn((3,))
    print(a)
    label = torch.tensor(1)
    print(label)

    print(F.cross_entropy(a,label))