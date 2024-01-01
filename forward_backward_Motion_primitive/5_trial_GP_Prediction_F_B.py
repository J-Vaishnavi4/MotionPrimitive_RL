#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv_forward import turtlebot3_burger_GymEnv_forward
from turtlebot3_burger_GymEnv_backward import turtlebot3_burger_GymEnv_backward

import datetime
from stable_baselines3 import ppo
from stable_baselines3.common.env_checker import check_env
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

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
    #check_env(env

    model = ppo.PPO.load(os.path.join(currentdir,"./best_models/PPO/translation_MP/"+MP_name+"2 (copy)"))

    GP_ = pickle.load(open(os.path.join(currentdir,"./GP_models/"+MP_name+"/noisy_exp_2_1.dump"), "rb"))
    required_displacement = 1.8
    mean_prediction, std_prediction = GP_.predict(np.array([required_displacement]).reshape(1,-1), return_std=True)
    print(mean_prediction[0])
    required_timesteps = round(mean_prediction[0][0])+1

    obs,info = env.reset()
    done = False
    rew=0
    ld, x, y = [info['ld']], [info['x']], [info['y']]

    for j in range(1000):

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
            print("Actual displacement: ",obs[0])
        env.render(mode='human')

    x_des = np.linspace(0,x[-1],len(x))
    y_des = x_des*math.tan(env._initial_orientation)
    plt.subplot(221)
    plt.plot(ld)
    plt.title('ld')
    plt.grid()
    plt.subplot(222)
    plt.plot(x,y, linestyle='dashed')
    plt.plot(x_des, y_des)
    plt.legend("actual path", "desired path")
    plt.title('path')
    plt.grid()
    plt.show()
    env.close()

if __name__ == '__main__':
  main()
