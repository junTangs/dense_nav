
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class ICM(nn.Module):
    def __init__(self,feature_extractor,forward_model,inverse_model,action_nums,beta = 0.5,alpha = 1):
        super(ICM, self).__init__()
        self.feature_extractor = feature_extractor
        self.forward_model  = forward_model
        self.inverse_model = inverse_model
        self.device = None

        self.action_nums = action_nums
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
        a_one_hot = F.one_hot(Variable(a).to(self.device),self.action_nums)
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


if __name__ == "__main__":
    a = torch.randn((3,))
    print(a)
    label = torch.tensor(1)
    print(label)

    print(F.cross_entropy(a,label))