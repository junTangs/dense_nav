import math
from abc import ABCMeta,abstractmethod
from crowd_sim.envs.utils.state import JointState,FullState,ObservableState
from crowd_sim.envs.utils.action import ActionRot,ActionXY
from nav_sim.utils.action import ActionXY as AXY
from nav_sim.utils.action import ActionVW as AVW

import configparser
from crowd_nav.policy.policy_factory import policy_factory
import torch

class NavAlg(metaclass = ABCMeta):
    def __init__(self,config_path,policy = 'cadrl') -> None:
        super().__init__()

        self.config = configparser.RawConfigParser()
        self.config.read(config_path)
        self.policy = policy_factory[policy]()
        self.device = None


    def setup(self,weight,time_step = 0.25,gpu = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
        self.policy.configure(config = self.config)

        self.policy.set_device(self.device)
        self.policy.get_model().load_state_dict(torch.load(weight))
        self.policy.set_phase("test")
        self.policy.time_step = time_step


    
    def preprocess(self,state)->JointState:

        rs = FullState(state[0],state[1],state[4],state[5],state[2],state[9],state[10],state[8],math.radians(state[3])+math.pi)

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
        state = self.preprocess(state)
        action = self.policy.predict(state)
        if isinstance(action,ActionXY):
            return AXY(action.vx,action.vy)
        else:
            return AVW(action.v,action.r)



    
    