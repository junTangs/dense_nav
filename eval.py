
import nav_sim
import gym
import json
from rl import rl_type,net_type
from colorama import Fore, Back, Style
import colorama
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

# args
EXPERT = "12s_21a_apf"
CONFIG_PATH = r'config/env_config/env.json'
NET_PATH = r'config/rl_config/q_net_config.json'
RL_PATH = r'config/rl_config/dqn_config.json'
MODEL_PATH = r"model/2022_05_14_21_42_12s_21a_apf/2500_ckpt"
EPISODE = int(100)


# load config
colorama.init(autoreset=True)
config = json.load(open(CONFIG_PATH))
net_config = json.load(open(NET_PATH))
rl_config = json.load(open(RL_PATH))

config["is_render"] = True

# environment
env = gym.make("nav_env-v0",config = config)

# reinforcement learning
net = net_type[net_config["net"]](net_config)
net2 = net_type[net_config["net"]](net_config)
rl = rl_type[rl_config["rl"]](rl_config)
rl.configure(target_net = net,eval_net = net2)
rl.load(MODEL_PATH)
rl.eval()

finish_counter = 0
ep_r_sum = 0
for i_episode in range(EPISODE):
    s = env.reset()

    step_cnt = 0
    ep_r = 0
    while True:
        a = rl.choose_action(s)
        s_,r,done,info  = env.step(a)

        rl.store(s,a,r,s_,done)

        ep_r += r
        step_cnt +=1
        s = s_


        env.render()
        if done or step_cnt >= 500:
            ep_r_sum+=ep_r
            if info["done_info"] == "finished":
                finish_counter += 1
                print(Back.GREEN +f"{i_episode}/{EPISODE} | info: {info}| reward: {ep_r}| average_reward:{ep_r/step_cnt}")
            else:
                print(f"{i_episode}/{EPISODE} | info: {info}| reward: {ep_r}| average_reward:{ep_r / step_cnt}")
            break

print(Back.RED + "reach_reate : {:.2f}| average_reward : {:3f}".format(finish_counter/EPISODE,ep_r_sum/EPISODE))






