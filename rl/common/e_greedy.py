import random
import torch
import numpy as np



def e_greedy(e:float,action_values:torch.Tensor)->int:
    rand = random.random()
    if rand <= e:
        return np.random.randint(0,action_values.shape[0])
    else:
        return int(torch.argmax(action_values))
    
    
if __name__ == "__main__":
    
    action_values = np.array([0.1,22,3,4])    
    action_values = torch.Tensor(action_values)
    a = e_greedy(0.005,action_values)
    print(a)