import os, inspect
import csv
import numpy as np
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from Gym_env_translation import translation_env
from Gym_env_rotation import rotation_env

import datetime
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
    
    # if algorithm == "1":
    #     algorithm = "PPO"
    #     model = ppo.PPO.load(os.path.join(currentdir,"./"+case+"/Policies/"+MP_name+"_"+model_num))
        
    # elif algorithm == "2":
    #     algorithm = "SAC"
    #     model = SAC.load(os.path.join(currentdir,"./"+case+"/"+MP_name+"/best_models/"+algorithm+"/diff_wheel_vel/"+str(i)))

    if not os.path.exists("./"+case+"/Samples"):
       os.makedirs("./"+case+"/Samples")

    
    "Action ([lin_vel, ang_vel]) according to the trained policy is applied for 'j' timesteps and then [0,0] action is applied.\
     Displacement (obs[0]) is observed at the end of (j+1)th timestep"
    model_numbers = [10]
    for model_num in model_numbers:
        if model_num == 10:
            j_range = 280
        elif model_num == 9:
            j_range = 600
        elif model_num == 8:
            j_range = 600
        elif model_num==7:
            j_range = 600
        elif model_num == 6:
            j_range = 310
        elif model_num == 5:
            j_range = 600
        elif model_num == 4:
            j_range = 600
        elif model_num == 3:
            j_range = 700
        elif model_num == 2:
            j_range = 880
        
        model = ppo.PPO.load(os.path.join(currentdir,"./"+case+"/Policies/"+MP_name+"_"+str(model_num)))

        obs,info = env.reset()
        with open("./"+case+"/Samples/"+MP_name+str(model_num)+'_samples.csv','w',newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["yaw_change", "time"])
        action = [0,0]
        data = obs[1]
        data = np.append(data, 0)
        with open("./"+case+"/Samples/"+MP_name+str(model_num)+'_samples.csv','a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            file.close()

        for j in range(j_range):
            obs,info = env.reset()
            for i in range(j+2):
                if i<=j:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done,truncated, info = env.step(action)
                else:
                    action = [0,0]
                    obs, reward, done,truncated, info = env.step(action)
                if i == j+1:
                    data = obs[1]
                    data = np.append(data, i)
                    with open("./"+case+"/Samples/"+MP_name+str(model_num)+'_samples.csv','a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(data)
                        file.close()
                env.render(mode='human')
    env.close()

if __name__ == '__main__':
  main()