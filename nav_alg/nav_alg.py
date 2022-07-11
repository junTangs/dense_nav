import math
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


    def setup(self,weight):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy.configure(self.config)
        self.policy.set_device(self.device)
        self.policy.get_model().load_state_dict(torch.load(weight))

    
    def preprocess(self,state)->JointState:
        vx = state[3]*math.cos(math.radians(state[5]))
        vy = state[3]*math.sin(math.radians(state[5]))
        v_pref = 1
        rs = FullState(state[0],state[1],vx,vy,state[2],state[6],state[7],v_pref,math.radians(state[5]))

        humans_states = state[8:]
        hs = []
        for i in range(int(len(humans_states)//7)):
            index = i*7
            hs.append(ObservableState(humans_states[index+1],
                                      humans_states[index+2],
                                      humans_states[index+3],
                                      humans_states[index+4],
                                      humans_states[index+5]))
        joint_state = JointState(rs,hs)
        return joint_state


    def predict(self,state):
        action = self.policy.predict(self.preprocess(state))
        return action['vx'],action['vy']


    
    