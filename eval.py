
import nav_sim
import gym
import json
from rl import QNet,rl_type
from colorama import Fore, Back, Style
import colorama
import os
import shutil
import datetime
from torch.utils.tensorboard import SummaryWriter
import copy
import argparse


parser = argparse.ArgumentParser(description="TrainNavAgent")
parser.add_argument("--env_config", type=str,default=r"config/env_config/env.json",help="env config file path")
parser.add_argument("--rl_config", type=str, default = r"config/rl_config/dqn_lstm_config.json",help="rl config file path")
parser.add_argument("--pre_trained",type = str,default=r"models/2022_06_02_10_00_v2_gru5_rgs_icm/5000_ckpt_73",help="pre trained")
parser.add_argument("--log_dir", type=str,default="log",help="log dir")
parser.add_argument("--episode",type= int,default=1e4,help="episode")
parser.add_argument("--name",type=str,default = "v2_gru_40_rgs_eval",help="experiment name")


args = parser.parse_args()

# args
EXPERT = args.name
CONFIG_PATH = args.env_config
RL_PATH = args.rl_config
LOG_DIR = args.log_dir
EPISODE = int(args.episode)

colorama.init(autoreset=True)
today =  datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_')

# load config
config = json.load(open(CONFIG_PATH))
config["is_render"] = True
rl_config = json.load(open(RL_PATH))
rl_config["pre_trained"] = args.pre_trained

# mkdir and save log
# log_dir = os.path.join(LOG_DIR,today+EXPERT)
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)

# log
# writer = SummaryWriter(log_dir=log_dir)

# environment
env = gym.make("nav_env-v0",config = config)

# reinforcement learning
rl = rl_type[rl_config["rl"]](rl_config)
rl.eval()

reach_counter = 0
for i_episode in range(EPISODE):
    s = env.reset()


    if i_episode%100 == 0 and i_episode!= 0:
        print(Back.BLUE + f" lastest reach rate : {reach_counter}")
        # writer.add_scalar("Eval/reach_rate",reach_counter,i_episode)
        reach_counter = 0

    step_cnt = 0
    ep_r = 0
    while True:
        a = rl.choose_action(s)

        s_,r,done,info  = env.step(a)
        env.render()

        ep_r += r
        step_cnt +=1
        s = s_


        env.render()


        if done or step_cnt >= 500:
            # writer.add_scalar("Eval/reward",ep_r,i_episode)
            # writer.add_scalar("Eval/steps",step_cnt,i_episode)
            # writer.add_scalar("Eval/average_reward",ep_r/step_cnt,i_episode)
            if info["done_info"] == "finished":
                reach_counter+=1
                print(Back.GREEN +f"{i_episode}/{EPISODE} | info: {info}| reward: {ep_r}| average_reward:{ep_r/step_cnt}")
            else:
                print(f"{i_episode}/{EPISODE} | info: {info}| reward: {ep_r}| average_reward:{ep_r / step_cnt}")
            break






