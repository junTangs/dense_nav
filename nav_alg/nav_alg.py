from abc import ABCMeta,abstractmethod
from .state import JointState,FullState,ObservableState
from .action import ActionRot,ActionXY
import configparser
from crowd_nav.policy.policy_factory import policy_factory
import torch

class NavAlg(metaclass = ABCMeta):
    def __init__(self,config_path,policy = 'cadrl') -> None:
        super().__init__()

        self.config = configparser.RawConfigParser()
        self.config.read(config_path)
        self.policy = policy_factory[policy]
        self.device = None
    

    def setup(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        self.policy.configure(self.config)
        self.policy.set_device(self.device)

    
    def preprocess(self,state)->JointState:
        """transfern state to joint state

        Args:
            state (list): frames state
        """

        raise NotImplementedError


    def predict(self,state):
        action = self.policy.predict(self.preprocess(state))
        return action['vx'],action['vy']


    
    