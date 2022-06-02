
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
parser.add_argument("--save_dir", type=str,default="models",help="save dir")
parser.add_argument("--log_dir", type=str,default="log",help="log dir")
parser.add_argument("--episode",type= int,default=1e4,help="episode")
parser.add_argument("--name",type=str,default = "v2_gru5_20",help="experiment name")


args = parser.parse_args()

# args
EXPERT = args.name
CONFIG_PATH = args.env_config
RL_PATH = args.rl_config
SAVE_DIR = args.save_dir
LOG_DIR = args.log_dir
EPISODE = int(args.episode)

colorama.init(autoreset=True)
today =  datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_')

# load config
config = json.load(open(CONFIG_PATH))
rl_config = json.load(open(RL_PATH))

# mkdir and save config
save_dir = os.path.join(SAVE_DIR,today+EXPERT)
save_config_dir = os.path.join(save_dir,"config")
save_env_config_dir = os.path.join(save_config_dir,"env_config")
save_rl_config_dir = os.path.join(save_config_dir,"rl_config")
save_config = copy.deepcopy(config)
save_rl_config = copy.deepcopy(rl_config)

save_config["robot_config_path"] = os.path.join(save_env_config_dir,os.path.basename(config["robot_config_path"]))
save_config["human_config_path"] = os.path.join(save_env_config_dir,os.path.basename(config["human_config_path"]))
save_config["obstacle_config_path"] = os.path.join(save_env_config_dir,os.path.basename(config["obstacle_config_path"]))
save_config["goal_config_path"] = os.path.join(save_env_config_dir,os.path.basename(config["goal_config_path"]))

save_rl_config["net_config_path"] = os.path.join(save_rl_config_dir,os.path.basename(rl_config["net_config_path"]))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(save_config_dir)
    os.mkdir(save_env_config_dir)
    os.mkdir(save_rl_config_dir)
    
    json.dump(save_config,open(os.path.join(save_env_config_dir,"env.json"),"w"))
    shutil.copy(config["robot_config_path"],save_env_config_dir)
    shutil.copy(config["human_config_path"],save_env_config_dir)
    shutil.copy(config["obstacle_config_path"],save_env_config_dir)
    shutil.copy(config["goal_config_path"],save_env_config_dir)
    
    json.dump(save_rl_config,open(os.path.join(save_rl_config_dir,"rl.json"),"w"))
    shutil.copy(rl_config["net_config_path"],save_rl_config_dir)

# mkdir and save log
log_dir = os.path.join(LOG_DIR,today+EXPERT)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# log
writer = SummaryWriter(log_dir=log_dir)

# environment
env = gym.make("nav_env-v0",config = config)

# reinforcement learning
rl = rl_type[rl_config["rl"]](rl_config)


reach_counter = 0
for i_episode in range(EPISODE):
    s = env.reset()

    if i_episode%500 == 0 and i_episode != 0:
        dir = os.path.join(save_dir,f"{i_episode}_ckpt_{reach_counter}")
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






