#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(__file__)

parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import matplotlib.pyplot as plt

from turtlebot3_burger_GymEnv_forward import turtlebot3_burger_GymEnv_forward
from turtlebot3_burger_GymEnv_backward import turtlebot3_burger_GymEnv_backward
# from turtlebot3_burger_GymEnv_forward_4states import turtlebot3_burger_GymEnv_forward

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
    
    model = ppo.PPO.load(os.path.join(currentdir,"./models/PPO/translation_MP/"+MP_name+"13/99"))
    obs,info = env.reset()
    done = False
    rew=0
    rew1, rew2, reward_plot, ld,x,y, yaw_change = [info['rew1']], [info['rew2']], [0], [info['ld']], [info['x']], [info['y']],[info['yaw_change']]
    # displacement, yaw_change = [obs[0]], [obs[1]]
    displacement = [obs[0]]
    alpha = abs(env._initial_orientation - math.atan(info['y']/info['x']))
    print("test ini", env._initial_orientation)
    tan_theta = [obs[0]*math.sin(alpha)]
    

    for i in range(3000):
        action, _states = model.predict(obs, deterministic=True)
        # print("obs: ",obs)
        print("action: ",action)
        obs, reward, done,truncated, info = env.step(action)
        
        reward_plot.append(reward)
        rew1.append(info['rew1'])
        rew2.append(info['rew2'])
        ld.append(info['ld'])
        x.append(info['x'])
        y.append(info['y'])
        yaw_change.append(info['yaw_change'])
        alpha = abs(env._initial_orientation - math.atan(info['y']/info['x']))
        tan_theta.append(obs[0]*math.sin(alpha))

        displacement.append(obs[0])
        env.render(mode='human')
        rew+=reward
        if done:
            # obs,info = env.reset()
            # rew=0
            break
    print("rew: ", rew)
    x_des = np.linspace(0,x[-1],len(x))
    print("after for loop ", env._initial_orientation)
    y_des = x_des*math.tan(env._initial_orientation)
    env.close()
    plt.subplot(231)
    plt.plot(reward_plot)
    plt.title('total_reward')
    plt.grid()
    plt.subplot(232)
    plt.plot(x,y, linestyle='dashed')
    plt.plot(x_des, y_des)
    plt.legend("actual path", "desired path")
    plt.title('path')
    plt.grid()
    # plt.subplot(233)
    # plt.plot(tan_theta)
    # plt.title('ld_calc')
    # plt.grid()
    plt.subplot(234)
    plt.plot(ld)
    plt.title('ld')
    plt.grid()
    plt.subplot(235)
    plt.plot(displacement)
    plt.title('displacement')
    plt.grid()
    plt.subplot(236)
    plt.plot(yaw_change)
    plt.title('yaw_change')
    plt.grid()
    plt.show()
if __name__ == '__main__':
  main()
