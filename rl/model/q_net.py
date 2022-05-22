import torch.nn as nn
import torch
from rl.model.feature_model import FEATURE_EXTRACTOR_FACTORY
from rl.model.value_model import V_A_FACTORY
from rl.common.deep_model import MLP
from torch.autograd import Variable
from torch import FloatTensor

class QNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = None

        # network that extract feature
        self.feature_extractor = FEATURE_EXTRACTOR_FACTORY[config["feature"]["type"]](config["feature"])
        # network that cal v and a
        self.v_a_net = V_A_FACTORY[config["v_a"]["type"]](config["v_a"])

        self._is_configured = False


    def forward(self,states):
        return self.v_a_net(self.feature_extractor(states))


    def set_device(self,device):
        self.to(device)
        self.device = device
        self.feature_extractor.set_device(device)
        self.v_a_net.set_device(device)
        return


    def save(self,dir):
        self.feature_extractor.save(dir)
        self.v_a_net.save(dir)

    def load(self,dir):
        self.feature_extractor.load(dir)
        self.v_a_net.load(dir)



if __name__ == "__main__":
    import json
    net_config = json.load(open(r"C:\Users\Rcihard_Tang\Desktop\Nav\config\q_net_config.json"))
    net  = QNet(net_config)
    x = torch.randn((32,41))

    x = net(x)