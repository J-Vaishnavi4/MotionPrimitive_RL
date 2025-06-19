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
    case ="1"# input("1: Ideal case, 2: Diff wheel radius, 3: Diff max wheel velocity: ")
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
        MP_name = "Backward"
        env = translation_env(renders=True, isDiscrete=False, motion="Backward", case = case)
    
    if case == "1":
        case = "Ideal"
    elif case == "2":
        case = "Diff_wheel_radius"
    elif case == "3":
        case = "Diff_max_wheel_vel"
    else:
        raise SystemExit("Incorrect case number")
    
    if algorithm == "1":
        algo = "PPO"
        model = ppo.PPO.load(os.path.join(currentdir,"./"+case+"/Policies/F_10"))
        # model = ppo.PPO.load(os.path.join(currentdir,"./"+case+"/Policies/"+MP_name+"_"+str(3)))
        
    elif algorithm == "2":
        algo = "SAC"
        model = SAC.load(os.path.join(currentdir,"./"+case+"/"+MP_name+"/best_models/"+algo+"/diff_wheel_vel/"+str(i)))
    
    obs,info = env.reset()
    done = False
    total_rew=0
    # displacement, yaw_change = [obs[0]], [obs[1]]
    rew1, rew2, reward_plot, ld,x,y = [info['rew1']], [info['rew2']], [0], [info['ld']], [info['x']], [info['y']]
    displacement, yaw_change = [obs[0]], [info['yaw_change']]
    alpha = abs(env._initial_orientation - math.atan2(info['y'],info['x']))
    print("test ini", np.rad2deg(env._initial_orientation))
    tan_theta = [obs[0]*math.sin(alpha)]

    
    for i in range(6000):
        action, _states = model.predict(obs, deterministic=True)
        
        # print("action: ",action)
        # print(i)
        obs, reward, done,truncated, info = env.step(action)
        # print("obs: ", obs,  info['ld'], info['yaw_change'])
        reward_plot.append(reward)
        rew1.append(info['rew1'])
        rew2.append(info['rew2'])
        ld.append(info['ld'])
        x.append(info['x'])
        y.append(info['y'])
        alpha = abs(env._initial_orientation - math.atan2(info['y'],info['x']))
        tan_theta.append(obs[0]*math.sin(alpha))

        displacement.append(obs[0])
        yaw_change.append(info['yaw_change'])
        env.render(mode='human')
        total_rew+=reward
        if done:
            print("obs: ",obs[0], yaw_change[i], ld[-1], total_rew, rew2[-1])
            # time.sleep(5)
            # obs,info = env.reset()
            total_rew=0
            break
    # print("obs: ",obs[0], yaw_change[i], ld[-1], total_rew)
    env.close()

    # plt.rcParams.update({'font.size': 14})
    x_des = np.linspace(0,x[-1],len(x))
    # print("after for loop ", env._initial_orientation)
    y_des = x_des*math.tan(env._initial_orientation)
    plt.subplot(221)
    plt.plot(displacement, linewidth=2.0)
    plt.title("displacement")
    plt.xlabel('Timesteps')
    plt.ylabel('displacement (metres)')
    plt.grid()
    plt.subplot(222)
    plt.plot(yaw_change, linewidth=2.0)
    plt.title("yaw_change")
    plt.xlabel('Timesteps')
    plt.ylabel('yaw change (radians)')
    plt.grid()
    plt.subplot(223)
    plt.plot(ld, linewidth=2.0)
    plt.title("ld")
    plt.xlabel('Timesteps')
    plt.ylabel('ld (metres)')
    plt.grid()
    plt.subplot(224)
    plt.plot(x, y, x_des, y_des)
    plt.title("path")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()
if __name__ == '__main__':
  main()
