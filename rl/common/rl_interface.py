from abc import ABCMeta
import numpy as np



class RLInterface(metaclass = ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        
        
    
    def configure(self,*args,**kwargs):
        raise NotImplementedError    
    
    def choose_action(self,state: np.ndarray):
        raise NotImplementedError
    
    def store(self,state: np.ndarray,action,reward: float,next_state: np.ndarray,done: bool):
        raise NotImplementedError

    
    def update(self,*args,**kwargs):
        raise NotImplementedError
    
    def learn(self):
        raise NotImplementedError
    
    def save(self,save_dir):
        raise NotImplementedError
    
    def load(self,model_dir):
        raise NotImplementedError
