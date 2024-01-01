import os, inspect
import csv
import numpy as np
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv_forward import turtlebot3_burger_GymEnv_forward
from turtlebot3_burger_GymEnv_backward import turtlebot3_burger_GymEnv_backward

import datetime
from stable_baselines3 import ppo
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
    if not os.path.exists("./Samples/"+MP_name):
       os.makedirs("./Samples/"+MP_name)

    model = ppo.PPO.load(os.path.join(currentdir,"./best_models/PPO/translation_MP/"+MP_name+"2 (copy)"))

    obs,info = env.reset()
    done = False
    rew=0
    # rew1, rew2, rew3, reward_plot = [info['rew1']], [info['rew2']], [info['rew3']], [0]
    # displacement, yaw_change = [obs[0]], [obs[1]]

    with open("./Samples/"+MP_name+'/samples_2_1.csv','w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["displacement", "time"])
    action = [0,0]
    data = obs[0]
    data = np.append(data, 0)
    with open("./Samples/"+MP_name+'/samples_2_1.csv','a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
        file.close()
    "Action ([lin_vel, ang_vel]) according to the trained policy is applied for 'j' timesteps and then [0,0] action is applied.\
     Displacement (obs[0]) is observed at the end of (j+1)th timestep"

    for j in range(3000):
        obs,info = env.reset()
        for i in range(j+2):
            if i<=j:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done,truncated, info = env.step(action)

            else:
                action = [0,0]
                obs, reward, done,truncated, info = env.step(action)
            if i == j+1:
                data = obs[0]
                data = np.append(data, i)
                with open("./Samples/"+MP_name+'/samples_2_1.csv','a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
                    file.close()
            env.render(mode='human')
    env.close()

if __name__ == '__main__':
  main()
