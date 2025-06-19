#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from Gym_env_translation import translation_env
from Gym_env_rotation import rotation_env

import datetime
from stable_baselines3 import ppo, SAC
from stable_baselines3.common.env_checker import check_env
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math, time

def main():

    case = "1" #input("1: Ideal case, 2: Diff wheel radius, 3: Diff max wheel velocity: ")
    algorithm = "1" #input("1: PPO, 2: SAC: ")
    MP_name = "1"# input("1: CW, 2: CCW: 3: Forward, 4: Backward: ")
    if MP_name == "1":
        MP_name = "CW"
        env = rotation_env(renders=True, isDiscrete=False, motion="CW", case = case)
    elif MP_name == "2":
        MP_name = "CCW"
        env = rotation_env(renders=True, isDiscrete=False, motion="CCW", case = case)
    elif MP_name == "3":
        MP_name = "F"
        env = translation_env(renders=True, isDiscrete=False, motion="Forward", case = case)
    elif MP_name == "4":
        MP_name = "B"
        env = translation_env(renders=True, isDiscrete=False, motion="Backward", case = case)
    
    if case == "1":
        case = "Ideal"
    elif case == "2":
        case = "Diff_wheel_radius"
    elif case == "3":
        case = "Diff_max_wheel_vel"
    else:
        raise SystemExit("Incorrect case number")
    model_num = 10
    if algorithm == "1":
        algorithm = "PPO"
        model = ppo.PPO.load(os.path.join(currentdir,"./"+case+"/Policies/"+MP_name+"_"+str(model_num)))
        
    elif algorithm == "2":
        algorithm = "SAC"
        model = SAC.load(os.path.join(currentdir,"./"+case+"/"+MP_name+"/best_models/"+algorithm+"/diff_wheel_vel/"+str(10)))
    begin = time.time()
    GP_ = pickle.load(open("./"+case+"/GP_models/"+MP_name+str(model_num)+"_noise.dump", "rb"))
    print("time to load GP model: ", time.time()-begin)
    required_displacement = 2.5
    begin=time.time()
    mean_prediction, std_prediction = GP_.predict(np.array([required_displacement]).reshape(1,-1), return_std=True)
    print("time to predict: ", time.time()-begin)
    print(mean_prediction[0])
    required_timesteps = round(mean_prediction[0][0])

    obs,info = env.reset()
    done = False
    rew=0
    ld, x, y = [info['ld']], [info['x']], [info['y']]
    # init_time = time.time()
    for j in range(1100):

        if j <= required_timesteps:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done,truncated, info = env.step(action)
            ld.append(info['ld'])
            x.append(info['x'])
            y.append(info['y'])
        else:
            action = [0,0]
            obs, reward, done,truncated, info = env.step(action)
        if j == required_timesteps:
            print("Required displacement: "+ str(required_displacement)+"\nPredicted timesteps: "+str(required_timesteps)+ \
            "\nstandard deviation: "+str(std_prediction[0]))
            if MP_name == "F" or MP_name == "B":
                print("Actual displacement: ",obs[0])
            else:
                print("Actual yaw_change: ", obs[1])
            # print("Time taken: ",time.time()-init_time)
            
        env.render(mode='human')
    env.close()

if __name__ == '__main__':
  main()