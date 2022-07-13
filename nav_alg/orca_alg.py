
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import JointState,FullState,ObservableState
from crowd_sim.envs.utils.action import ActionXY,ActionRot
from nav_sim.utils.action import ActionXY as AXY
from nav_sim.utils.action import ActionVW as AVW
import math

class OrcaAlg:
    def __init__(self):
        self.orca = policy_factory['orca']()


    def setup(self,**kwargs):
        self.orca.__dict__.update(kwargs)
        return
    def preprocess(self, state) -> JointState:

        rs = FullState(state[0], state[1], state[4], state[5], state[2], state[9], state[10], state[8],
                       math.radians(state[3]) + math.pi)

        humans_states = state[8:]
        hs = []
        for i in range(int(len(humans_states) // 7)):
            index = i * 7
            hs.append(ObservableState(humans_states[index + 1],
                                      humans_states[index + 2],
                                      humans_states[index + 3],
                                      humans_states[index + 4],
                                      humans_states[index + 5]))
        joint_state = JointState(rs, hs)
        return joint_state

    def predict(self, state):
        state = self.preprocess(state)
        action = self.orca.predict(state)
        print(action)
        print(self.orca.__dict__)
        if isinstance(action, ActionXY):
            return AXY(action.vx, action.vy)
        else:
            return AVW(action.v, action.r)

    
    