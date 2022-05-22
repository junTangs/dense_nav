import torch.nn as nn
import torch
from abc import ABCMeta, abstractmethod
import os


class ValueAdvance(nn.Module, metaclass=ABCMeta):
    def __init__(self, config):
        super(ValueAdvance, self).__init__()
        self.config = config
        self.device = None

    @abstractmethod
    def v_a(self, x):
        raise NotImplementedError

    def forward(self, feature):
        return self.v_a(feature)

    def set_device(self,device):
        self.device = device
        self.to(device)
        return

    def save(self, dir):
        torch.save(self.state_dict(), os.path.join(dir, "v_a.pth"))

    def load(self, dir):
        self.load_state_dict(torch.load(os.path.join(dir, "v_a.pth")))



