import numpy as np
from collections import deque,namedtuple
import random



Transition = namedtuple("Transition",["s","a","r","s_","d"])
MiniBatch = namedtuple("MiniBatch",["s","a","r","s_","d"])

class ReplayBuffer:
    def __init__(self,config):
        self.config = config 
        
        self.capacity = config["capacity"]
        self.batch_size = config["batch_size"]        
        self.data = deque(maxlen=self.capacity)
        
        
        
    def sample(self):
        mini_batch = random.sample(self.data,self.batch_size)
        s_batch = np.array([transition.s for transition in mini_batch])
        a_batch = np.array([transition.a for transition in mini_batch]).reshape(self.batch_size,1)
        r_batch = np.array([transition.r for transition in mini_batch]).reshape(self.batch_size,1)
        s_prime_batch = np.array([transition.s_ for transition in mini_batch])
        d_batch = np.array([transition.d for transition in mini_batch]).reshape(self.batch_size,1)
        d_batch = np.where(d_batch==True,1,0)
    
        return MiniBatch(s_batch,a_batch,r_batch,s_prime_batch,d_batch)
        
    def store(self,state:np.ndarray,action,reward,next_state,done):
        if(len(self.data)>=self.capacity):
            self.data.popleft()
        transition = Transition(state,action,reward,next_state,done)
        self.data.append(transition)
        return 
    
    def __len__(self):
        return len(self.data)
    
    

        
if __name__ == "__main__":
    config = {"capacity":1000,"batch_size":256}
    replay_buffer = ReplayBuffer(config)
    
    for i in range(10000):
        replay_buffer.store(np.array([1,2,3]),i,2,np.array([4,5,6]),False)
        
    mini_batch = replay_buffer.sample()
    print(mini_batch)
    print(len(replay_buffer))
    