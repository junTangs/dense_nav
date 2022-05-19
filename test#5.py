
import nav_sim
import gym
import json
from rl import QNet,rl_type
from colorama import Fore, Back, Style
import colorama
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

# args
EXPERT = "s12_a21_apfm_0s_20h_v2l"
CONFIG_PATH = r'config/env_manhattan_config/env.json'
NET_PATH = r'config/rl_config/q_net_v2_light.json'
RL_PATH = r'config/rl_config/dqn_config.json'
SAVE_DIR = "model"
LOG_DIR = "log"
EPISODE = int(1e4)

colorama.init(autoreset=True)
today =  datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_')
save_dir = os.path.join(SAVE_DIR,today+EXPERT)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

log_dir = os.path.join(LOG_DIR,today+EXPERT)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# load config
config = json.load(open(CONFIG_PATH))
net_config = json.load(open(NET_PATH))
rl_config = json.load(open(RL_PATH))

# log
writer = SummaryWriter(log_dir=log_dir)

# environment
env = gym.make("nav_env-v0",config = config)

# reinforcement learning
net = QNet(net_config)
net2 = QNet(net_config)
rl = rl_type[rl_config["rl"]](rl_config)
rl.configure(target_net = net,eval_net = net2)


reach_counter = 0
for i_episode in range(EPISODE):
    s = env.reset()

    if i_episode%500 == 0 and i_episode != 0:
        dir = os.path.join(save_dir,f"{i_episode}_ckpt")
        os.mkdir(dir)
        rl.save(dir)

    if i_episode%100 == 0 and i_episode!= 0:
        print(Back.BLUE + f" lastest reach rate : {reach_counter}")
        writer.add_scalar("Train/reach_rate",reach_counter,i_episode)
        reach_counter = 0

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

        loss = rl.learn()
        writer.add_scalar("Train/loss", loss, i_episode)

        if done or step_cnt >= 500:
            writer.add_scalar("Train/reward",ep_r,i_episode)
            writer.add_scalar("Train/steps",step_cnt,i_episode)
            writer.add_scalar("Train/average_reward",ep_r/step_cnt,i_episode)
            if info["done_info"] == "finished":
                reach_counter+=1
                print(Back.GREEN +f"{i_episode}/{EPISODE} | info: {info}| reward: {ep_r}| average_reward:{ep_r/step_cnt}")
            else:
                print(f"{i_episode}/{EPISODE} | info: {info}| reward: {ep_r}| average_reward:{ep_r / step_cnt}")
            break






