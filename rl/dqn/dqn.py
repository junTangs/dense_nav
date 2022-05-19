
from ..common.relpay_buffer import ReplayBuffer
from ..common.rl_interface import RLInterface
from ..common.e_greedy import e_greedy

from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
from torch import FloatTensor,LongTensor
import torch
import numpy as np

class DQN(RLInterface):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config 
        
        # replay buffer
        self.memory = ReplayBuffer(config["replay_buffer"])
        
        # model 
        self.target_net = None
        self.eval_net = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        
        self.gamma = config["gamma"]
        self.e = config["e"]
        self.update_freq = config["update_freq"]
        self.lr = config["lr"]
        
        
        # configure
        self.is_configured = False
        self.use_gpu = config["use_gpu"]
        
        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
            
        self.loss_fn.to(self.device)
        self.learn_step = 0
            
            
    
    def configure(self, *args, **kwargs):
        self.target_net = kwargs["target_net"]
        self.eval_net = kwargs["eval_net"]
        self.optimizer = Adam(self.eval_net.parameters(), lr=self.lr)

        self.target_net.to(self.device)
        self.eval_net.to(self.device)
        
        self.is_configured = True
        return 
    
    def choose_action(self, state: np.ndarray):
        assert(self.is_configured), "[DQN]:DQN is not configured"
        # [*s_dim] -> [1,*s_dim]
        s = Variable(FloatTensor(state)).to(self.device).reshape(1,*state.shape)
        # [1,a_dim] -> [a_dim]
        q = self.eval_net(s).reshape(-1)
        
        a = e_greedy(self.e,q)
        return a
    

    def store(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool):
        self.memory.store(state,action,reward,next_state,done)
        return
    
    @property
    def memory_size(self):
        return len(self.memory)
    
    def update(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
        return 

    def learn(self):
        assert(self.is_configured), "[DQN]:DQN is not configured"
        
        if self.learn_step % self.update_freq == 0:
            self.update()
        
        # mini_batch
        mini_batch = self.memory.sample()     
        
        # [b,*s_dim]
        s = Variable(FloatTensor(mini_batch.s)).to(self.device)
        # [b,1] 
        a = Variable(LongTensor(mini_batch.a)).to(self.device)
        # [b,1]
        r  = Variable(FloatTensor(mini_batch.r)).to(self.device)
        # [b,*s_dim]
        s_ = Variable(FloatTensor(mini_batch.s_)).to(self.device)
        # [b,1]
        d = Variable(LongTensor(mini_batch.d)).to(self.device)
        
        # Q(s) : [b,a_dim]
        q_s = self.eval_net(s)
        
        # Q(s,a): [b,1]
        q_s_a = q_s.gather(1,a)
        
        
        # a' : [b,1]
        a_  = torch.argmax(q_s,1).reshape(-1,1)
        # Q'(s',a') =Q'(s_,argmax_a(Q(s,a))) : [b,1]
        qt_s_a_ = self.target_net(s_).gather(1,a_)
        
        y = r + self.gamma*qt_s_a_*(1-d)
        # y = r + self.gamma*qt_s_a_

        loss = self.loss_fn(y,q_s_a)
        
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learn_step += 1
        
        return float(loss)
    
    def save(self, save_dir):
        torch.save(self.eval_net.state_dict(), save_dir+"/eval_net.pth")
        torch.save(self.target_net.state_dict(), save_dir+"/target_net.pth")
        return
    
    def load(self, model_dir):
        self.eval_net.load_state_dict(torch.load(model_dir+"/eval_net.pth"))
        self.target_net.load_state_dict(torch.load(model_dir+"/target_net.pth"))
        return


    def train(self):
        self.e = self.config["e"]
        self.target_net.train()
        self.eval_net.train()


    def eval(self):
        self.e = 0
        self.target_net.eval()
        self.eval_net.eval()

        
        
        
        
          