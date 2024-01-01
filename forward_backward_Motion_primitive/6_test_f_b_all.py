#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(__file__)

parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import matplotlib.pyplot as plt

from turtlebot3_burger_GymEnv_forward import turtlebot3_burger_GymEnv_forward
from turtlebot3_burger_GymEnv_backward import turtlebot3_burger_GymEnv_backward
from turtlebot3_burger_GymEnv_forward_4states import turtlebot3_burger_GymEnv_forward

import math
import numpy as np
from stable_baselines3 import ppo, SAC
from stable_baselines3.common.env_checker import check_env

def main():
    MP_name = input("Forward(F)/ Backward (B) MP: ")
    if MP_name == "F":
        MP_name = "Forward"
        env = turtlebot3_burger_GymEnv_forward(renders=True, isDiscrete=False)
    elif MP_name == "B":
        MP_name = "Backward"
        env = turtlebot3_burger_GymEnv_backward(renders=True, isDiscrete=False)
    else:
        raise SystemExit("Incorrect MP name")
    #check_env(env)
    displacement_list = []
    for j in range(80,97):
        model = ppo.PPO.load(os.path.join(currentdir,"./models/PPO/translation_MP/"+MP_name+"7/"+str(j)))
        obs,info = env.reset()
        done = False
        # rew=0
        # rew1, rew2, reward_plot, ld,x,y, yaw_change = [info['rew1']], [info['rew2']], [0], [info['ld']], [info['x']], [info['y']],[info['yaw_change']]
        # displacement = [obs[0]]
        # alpha = abs(env._initial_orientation - math.atan(info['y']/info['x']))
        print("test ini", env._initial_orientation)
        # tan_theta = [obs[0]*math.sin(alpha)]
        

        for i in range(3000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done,truncated, info = env.step(action)
            env.render(mode='human')
            if done:
                break
        displacement_list.append(obs[0])
    print(displacement_list)
if __name__ == '__main__':
  main()
