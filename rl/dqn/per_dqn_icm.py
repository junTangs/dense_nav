
from re import S

from sympy import im
from ..common.priority_relpay_buffer import ReplayBuffer
from ..common.rl_interface import RLInterface
from ..common.e_greedy import e_greedy
from rl.common.icm import ICM,ForwardModel,InverseModel
from rl import QNet
import copy
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
from torch import FloatTensor,LongTensor
import torch
import numpy as np
from rl.common.soft_update import soft_update
import itertools
import json
import os
class PER_DQN_ICM(RLInterface):
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
        self.use_intrinsic = config["use_intrinsic"]
        self.use_td_clip = config["use_td_clip"]
        
        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
            
        self.loss_fn.to(self.device)
        self.learn_step = 0


        # icm
        self.icm = None
        
        self.net_config = json.load(open(config["net_config_path"]))
        
        self.target_net = QNet(self.net_config)
        self.eval_net = QNet(self.net_config)

        if self.use_intrinsic:
            feature_extractor = self.net_config["feature_extractor"]
            self.icm = ICM(feature_extractor,self.config["feature_dims"],self.config["action_dims"])
            self.optimizer = Adam(itertools.chain(self.eval_net.parameters(), self.icm.parameters()), lr=self.lr)
        else:
            self.optimizer = Adam(self.eval_net.parameters(), lr=self.lr)

        self.target_net.set_device(self.device)
        self.eval_net.set_device(self.device)

            
            
    
    def configure(self, *args, **kwargs):
        self.target_net = kwargs["target_net"]
        self.eval_net = kwargs["eval_net"]

        if self.use_intrinsic:
            feature_extractor = copy.deepcopy(QNet.feature_extractor)
            self.icm = ICM(feature_extractor,)
            self.optimizer = Adam(itertools.chain(self.eval_net.parameters(), self.icm.parameters()), lr=self.lr)
        else:
            self.optimizer = Adam(self.eval_net.parameters(), lr=self.lr)

        self.target_net.set_device(self.device)
        self.eval_net.set_device(self.device)

        self.is_configured = True
        return 
    
    def choose_action(self, state,**kwargs):
        assert(self.is_configured), "[DQN]:DQN is not configured"
        q = self.eval_net([state])
        
        a = e_greedy(self.e,q)
        return a

    

    def store(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool):
        self.memory.store(state,action,reward,next_state,done)
        return
    
    @property
    def memory_size(self):
        return len(self.memory)
    
    def update(self):
        soft_update(target=self.target_net,source=self.eval_net,x = 0.9)
        return 

    def learn(self):
        assert(self.is_configured), "[DQN]:DQN is not configured"

        if(len(self.memory) <= self.config["replay_buffer"]["batch_size"]):
            return 0
        
        if self.learn_step % self.update_freq == 0:
            self.update()
        
        # mini_batch
        batch_index,mini_batch,ISWeight = self.memory.sample()
        
        # b*[*s_dim]
        s = mini_batch.s

        # [b,1] 
        a = Variable(LongTensor(mini_batch.a)).to(self.device)
        # [b,1]
        r  = Variable(FloatTensor(mini_batch.r)).to(self.device)
        # b*[*s_dim]
        s_ = mini_batch.s_
        # [b,1]
        d = Variable(LongTensor(mini_batch.d)).to(self.device)

        ISWeight = Variable(FloatTensor(ISWeight)).to(self.device)
        
        # Q(s) : [b,a_dim]
        q_s = self.eval_net(s)
        
        # Q(s,a): [b,1]
        q_s_a = q_s.gather(1,a)

        if self.use_intrinsic:
            intrinsic_r,total_loss,_,_ = self.icm(s,s_,a)
            r = r + intrinsic_r
        
        
        
        
        # a' : [b,1]
        a_  = torch.argmax(q_s,1).reshape(-1,1)
        # Q'(s',a') =Q'(s_,argmax_a(Q(s,a))) : [b,1]
        qt_s_a_ = self.target_net(s_).gather(1,a_)
        
        # y = r + self.gamma*qt_s_a_*(1-d)
        y = r + self.gamma*qt_s_a_*(1-d)

        # [b,1]
        td_errors = abs(y - q_s_a)
        
        td_errors = torch.clamp(td_errors,0,1) if self.use_td_clip else td_errors

        loss = torch.mean(ISWeight*(td_errors)**2)

        loss += total_loss if self.use_intrinsic else 0
        
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        td_errors = td_errors.cpu().detach().numpy()
        self.memory.batch_update(batch_index,td_errors)
        self.learn_step += 1
        
        return float(loss)
    
    def save(self, save_dir):
        torch.save(self.eval_net.state_dict(), save_dir+"/eval_net.pth")
        torch.save(self.target_net.state_dict(), save_dir+"/target_net.pth")
        self.memory.save(save_dir+"/memory.pkl")
        return
    
    def load(self, model_dir):
        self.eval_net.load_state_dict(torch.load(model_dir+"/eval_net.pth"))
        self.target_net.load_state_dict(torch.load(model_dir+"/target_net.pth"))
        
        if os.path.exists(model_dir+"/memory.pkl"):
            self.memory.load(model_dir+"/memory.pkl")
        return
    
    def load_memory(self,memory_path):
        self.memory.load(memory_path)
        return 
    
    def save_memory(self,memory_path):
        self.memory.save(memory_path)
        return 

    def train(self):
        self.e = self.config["e"]
        self.target_net.train()
        self.eval_net.train()


    def eval(self):
        self.e = 0
        self.target_net.eval()
        self.eval_net.eval()
        
        
        
        
          