import torch.nn as nn
import torch
from abc import ABCMeta,abstractmethod
import os

class FeatureExtractor(nn.Module,metaclass=ABCMeta):
    '''
    Feature Extractor
    '''
    def __init__(self,config):
        super(FeatureExtractor, self).__init__()
        self.config = config
        self.device = None

    @abstractmethod
    def preprocess(self,states):
        # preprocess [ state_dic_0 , state_dic_1]
        raise NotImplementedError

    @abstractmethod
    def feature(self,*args,**kwargs):
        raise NotImplementedError

    def forward(self,states):
        return self.feature(self.preprocess(states))

    def set_device(self,device):
        self.device = device
        self.to(device)
        return

    def save(self,dir):
        torch.save(self.state_dict(),os.path.join(dir,"extractor.pth"))

    def load(self,dir):
        self.load_state_dict(torch.load(os.path.join(dir,"extractor.pth")))









