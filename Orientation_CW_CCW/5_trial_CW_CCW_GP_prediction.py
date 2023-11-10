#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv_CCW import turtlebot3_burger_GymEnv_CCW
from turtlebot3_burger_GymEnv_CW import turtlebot3_burger_GymEnv_CW
import datetime
from stable_baselines3 import ppo
from stable_baselines3.common.env_checker import check_env
import pickle
import numpy as np

def main():

    MP_name = input("Clockwise(CW)/ Counter Clockwise (CCW) MP: ")
    if MP_name == "CW":
        env = turtlebot3_burger_GymEnv_CW(renders=True, isDiscrete=False)
    elif MP_name == "CCW":
        env = turtlebot3_burger_GymEnv_CCW(renders=True, isDiscrete=False)
    else:
        raise SystemExit("Incorrect MP name")
    #check_env(env

    model = ppo.PPO.load(os.path.join(currentdir,"./best_models/PPO/orientation_MP_"+MP_name))

    GP_ = pickle.load(open(os.path.join(currentdir,"./GP_models/"+MP_name+"/no_noise.dump"), "rb"))
    required_yaw = 1.5
    mean_prediction, std_prediction = GP_.predict(np.array([required_yaw]).reshape(1,-1), return_std=True)
    required_timesteps = round(mean_prediction[0])-1
    print("Orientation_change: "+ str(required_yaw)+"\ntimesteps: "+str(required_timesteps)+ "\nstandard deviation"+str(std_prediction[0]))

    obs,info = env.reset()
    done = False
    rew=0

    for j in range(100):

        if j <= required_timesteps:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done,truncated, info = env.step(action)

        else:
            action = [0,0]
            obs, reward, done,truncated, info = env.step(action)
        if j == required_timesteps:
            data = obs[1]
            data = np.append(data, j+1)
            print(data)
        env.render(mode='human')

    env.close()

if __name__ == '__main__':
  main()
