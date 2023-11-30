#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(__file__)

parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import matplotlib.pyplot as plt

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

    model = ppo.PPO.load(os.path.join(currentdir,"./models/PPO/translation_MP/"+MP_name+"3 (copy)"))

    obs,info = env.reset()
    done = False
    rew=0
    rew1, rew2, reward_plot = [info['rew1']], [info['rew2']], [0]
    displacement, yaw_change = [obs[0]], [obs[1]]

    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done,truncated, info = env.step(action)
        reward_plot.append(reward)
        rew1.append(info['rew1'])
        rew2.append(info['rew2'])
        # rew3.append(info['rew3'])
        print(action[1])
        displacement.append(obs[0])
        yaw_change.append(obs[1])
        env.render(mode='human')
        rew+=reward
        if done:
            obs,info = env.reset()
            print("done: ", rew)
            rew=0
            break

    env.close()
    plt.subplot(231)
    plt.plot(reward_plot)
    plt.title('total_reward')
    plt.grid()
    plt.subplot(232)
    plt.plot(rew1)
    plt.title('rew1')
    plt.grid()
    plt.subplot(233)
    plt.plot(rew2)
    plt.title('rew2')
    plt.grid()
    # plt.subplot(234)
    # plt.plot(rew3)
    # plt.title('rew3')
    # plt.grid()
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
