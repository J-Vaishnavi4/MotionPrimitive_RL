#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import matplotlib.pyplot as plt

from Gym_env_translation import translation_env
from Gym_env_rotation import rotation_env

import math, time, csv
import numpy as np
from stable_baselines3 import ppo, SAC
from stable_baselines3.common.env_checker import check_env

def main():
    case = input("1: Ideal case, 2: Diff wheel radius, 3: Diff max wheel velocity: ")
    algorithm = "1" #input("1: PPO, 2: SAC: ")
    MP = input("1: CW, 2: CCW: 3: Forward, 4: Backward: ")
    pts = np.arange(-180,180,1)
    if case == "1":
        case_name = "Ideal"
    elif case == "2":
        case_name = "Diff_wheel_radius"
    elif case == "3":
        case_name = "Diff_max_wheel_vel"
    else:
        raise SystemExit("Incorrect case number")
    old_model_reward = 0
    model_numbers = [912,914,954,960,961,965,966,983,658,659,66,748,771,772,774,775,776,843,845,846,850,867,868,869,870,883,900,901,902,904,911]
    for model_num in model_numbers:
        fail = 0
        n=0
        for k in pts:
            # print(k, "degree")
            if MP == "1":
                MP_name = "CW"
                env = rotation_env(renders=True, isDiscrete=False, motion="CW", case = case, init_angle = np.radians(k))
            elif MP == "2":
                MP_name = "CCW"
                env = rotation_env(renders=True, isDiscrete=False, motion="CCW", case = case, init_angle = np.radians(k))
            elif MP == "3":
                MP_name = "Forward"
                env = translation_env(renders=True, isDiscrete=False, motion="Forward", case = case, init_angle = np.radians(k))
            elif MP == "4":
                MP_name = "Backward"
                env = translation_env(renders=True, isDiscrete=False, motion="Backward", case = case, init_angle = np.radians(k))
            
            
            
            if algorithm == "1":
                # print("model_num: ", model_num)
                algo = "PPO"
                model = ppo.PPO.load(os.path.join(currentdir,"./"+case_name+"/"+MP_name+"/models/"+algo+"/model_6/"+str(model_num)))
                
            elif algorithm == "2":
                algo = "SAC"
                model = SAC.load(os.path.join(currentdir,"./"+case_name+"/"+MP_name+"/best_models/"+algo+"/diff_wheel_vel/"+str(i)))
            obs,info = env.reset()
            done = False
            for i in range(10000):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done,truncated, info = env.step(action)
                env.render(mode='human')
                if MP_name == "Forward" or MP_name == "Backward":
                    if done or obs[0]>0.4:
                        
                        if obs[0] < 0.3:
                            fail += 1
                            n=0
                            print("Fail ",fail,k, obs[0])
                            if obs[0]<0.2:
                                fail = 5
                        else:
                            n=1
                        break
                else:# MP_name == "CW" or MP_name == "CCW":
                    if done:
                        # print("yaw change: ", obs[1])
                        if obs[1] < math.pi:
                            fail += 1
                            print("Fail ",fail,k, obs[1])
                            n=0
                        else:
                            n=1
                        break
            if fail == 5:
                break
        if n == 1:# and k==179:
            print("correct RL Model : ", model_num, "failed: ", fail)
            # if reward > old_model_reward:
            with open(os.path.join(currentdir,"./"+case_name+"/"+MP_name+"/models/")+'file.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([model_num, fail])
            old_model_reward = reward
        else:
            print(model_num, "doesn't work", "failed: ", fail)

if __name__ == '__main__':
  main()
