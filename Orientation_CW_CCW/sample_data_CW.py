#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
import csv
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv_CW import turtlebot3_burger_GymEnv_CW
import datetime
from stable_baselines3 import ppo
from stable_baselines3.common.env_checker import check_env

def main():

    env = turtlebot3_burger_GymEnv_CW(renders=True, isDiscrete=False)
    # It will check your custom environment and output additional warnings if needed
    #check_env(env)

    # model = ppo.PPO("MlpPolicy", env, verbose=1)
    # print("############Training completed################")
    model = ppo.PPO.load(os.path.join(currentdir,"./best_models/PPO/orientation_MP_CW"))

    obs,info = env.reset()
    done = False
    rew=0
    with open('CW_samples1.csv','w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Lin_Vel", "Ang_Vel","Yaw_change", "time"])
    action = [0,0]
    data = action
    data = np.append(data, obs[1])
    data = np.append(data, 0)
    with open('CW_samples1.csv','a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
        file.close()

    for j in range(55):
        obs,info = env.reset()
        for i in range(j+10):
            if i<=j:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done,truncated, info = env.step(action)

            else:
                action = [0,0]
                obs, reward, done,truncated, info = env.step(action)
            if i == j:
                data = action
                data = np.append(data, obs[1])
                data = np.append(data, j+1)
                with open('CW_samples1.csv','a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
                    file.close()
            env.render(mode='human')

    env.close()

if __name__ == '__main__':
  main()
