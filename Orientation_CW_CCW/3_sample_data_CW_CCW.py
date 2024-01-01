#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
import csv
import numpy as np
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv_CW import turtlebot3_burger_GymEnv_CW
from turtlebot3_burger_GymEnv_CCW import turtlebot3_burger_GymEnv_CCW
import datetime
from stable_baselines3 import ppo
from stable_baselines3.common.env_checker import check_env

def main():
    MP_name = input("Clockwise(CW)/ Counter Clockwise (CCW) MP: ")
    if MP_name == "CW":
        env = turtlebot3_burger_GymEnv_CW(renders=True, isDiscrete=False)
    elif MP_name == "CCW":
        env = turtlebot3_burger_GymEnv_CCW(renders=True, isDiscrete=False)
    else:
        raise SystemExit("Incorrect MP name")
    #check_env(env
    if not os.path.exists("./Samples/"+MP_name):
       os.makedirs("./Samples/"+MP_name)

    # samples_21_90 for CCW, samples_21_70 for CW

    # model = ppo.PPO.load(os.path.join(currentdir,"./models/PPO/orientation_MP/"+MP_name+"21/90"))     #CCW
    model = ppo.PPO.load(os.path.join(currentdir,"./models/PPO/orientation_MP/"+MP_name+"21/70"))       #CW
    obs,info = env.reset()
    done = False
    rew=0
    with open("./Samples/"+MP_name+'/samples_21_90.csv','w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Yaw_change", "time"])
    action = [0,0]
    data = obs[1]
    data = np.append(data, 0)
    with open("./Samples/"+MP_name+'/samples_21_90.csv','a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
        file.close()

    "Action ([lin_vel, ang_vel]) according to the trained policy is applied for 'j' timesteps and then [0,0] action is applied.\
     Yaw change (obs[1]) is observed at the end of (j+1)th timestep"

    for j in range(350):
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
                with open("./Samples/"+MP_name+'/samples_21_90.csv','a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
                    file.close()
            env.render(mode='human')

    env.close()

if __name__ == '__main__':
  main()
