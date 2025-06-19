#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import matplotlib.pyplot as plt

from Gym_env_translation import translation_env
from Gym_env_rotation import rotation_env

import math, time
import numpy as np
from stable_baselines3 import ppo, SAC
from stable_baselines3.common.env_checker import check_env

def main():
    case = "1" #input("1: Ideal case, 2: Diff wheel radius, 3: Diff max wheel velocity: ")
    algorithm = "1" #input("1: PPO, 2: SAC: ")
    MP_name = input("1: CW, 2: CCW: 3: Forward, 4: Backward: ")
    if MP_name == "1":
        MP_name = "CW"
        env = rotation_env(renders=True, isDiscrete=False, motion="CW", case = case)
    elif MP_name == "2":
        MP_name = "CCW"
        env = rotation_env(renders=True, isDiscrete=False, motion="CCW", case = case)
    elif MP_name == "3":
        MP_name = "Forward"
        env = translation_env(renders=True, isDiscrete=False, motion="Forward", case = case)
    elif MP_name == "4":
        MP_name = "Backward"
        env = translation_env(renders=True, isDiscrete=False, motion="Backward", case = case)
    
    if case == "1":
        case_name = "Ideal"
    elif case == "2":
        case_name = "Diff_wheel_radius"
    elif case == "3":
        case_name = "Diff_max_wheel_vel"
    else:
        raise SystemExit("Incorrect case number")
    
    for j in range(900,1000):
        if algorithm == "1":
            algorithm = "PPO"
            model = ppo.PPO.load(os.path.join(currentdir,"./"+case_name+"/"+MP_name+"/models/"+algorithm+"/model_12/"+str(j)))
        
        elif algorithm == "2":
            algorithm = "SAC"
            model = SAC.load(os.path.join(currentdir,"./"+case_name+"/"+MP_name+"/models/"+algorithm+"/diff_wheel_vel/"+str(j)))

        n=0
        for num in range(10):
            obs,info = env.reset()
            done = False

            for i in range(4000):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done,truncated, info = env.step(action)
                # print("obs ", obs)
                env.render(mode='human')
                if MP_name == "Forward" or MP_name == "Backward":
                    if done or obs[0]>1:
                        print("displacement: ", obs[0], info['ld'], info['yaw_change'])
                        if obs[0] < 0.50:
                            break
                        else:
                            n+=1
                        break
                else:# MP_name == "CW" or MP_name == "CCW":
                    if done:
                        # print("yaw change: ", obs[1])
                        if obs[1] < math.pi:
                            break
                        else:
                            n+=1
                        break
                

            if num-n>=2:
                break
        if n>7:
            print("model ok", n, j)
            print("displacement", obs[0])
        else:
            print("model not ok",n,j, "displacement: ", obs[0])

    env.close()

if __name__ == '__main__':
  main()
