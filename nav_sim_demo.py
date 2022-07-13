import matplotlib.pyplot as plt

from nav_sim.envs import NavEnvV1,NavEnvV2
import json
import tqdm
from pprint import pprint
from nav_alg import NavAlg,OrcaAlg

config_path = r'config/env_config/env.json'
config = json.load(open(config_path))
env = NavEnvV1(config)
s = env.reset()

alg = OrcaAlg()
alg.setup(time_step = 0.01,max_neighbors=20,radius= 0.25,max_speed = 1,)

for i in tqdm.trange(10000):
    a = alg.predict(s)
    s_, r, done, info = env.step(a)
    env.render()
    s = s_
    if done:
        s = env.reset()


