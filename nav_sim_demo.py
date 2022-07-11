import matplotlib.pyplot as plt

from nav_sim.envs import NavEnvV1
import json
import tqdm
from pprint import pprint
from nav_alg import NavAlg

config_path = r'config/env_config/env_demo.json'
config = json.load(open(config_path))
env = NavEnvV1(config)
env.reset()

cadrl = NavAlg(".")

for i in tqdm.trange(10000):
    s = env.step(env.action_space[1])
    print(cadrl.preprocess(s[0]).__dict__)
    print(s)
    env.render()


