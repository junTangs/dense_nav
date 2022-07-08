import matplotlib.pyplot as plt

from nav_sim.envs import NavEnvV1
import json
import tqdm
from pprint import pprint

config_path = r'config/env_config/env_demo.json'
config = json.load(open(config_path))
env = NavEnvV1(config)
env.reset()

for i in tqdm.trange(10000):
    s = env.step(0)
    print(s)
    env.render()


