#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv_CW import turtlebot3_burger_GymEnv_CW
from turtlebot3_burger_GymEnv_CCW import turtlebot3_burger_GymEnv_CCW
import matplotlib.pyplot as plt

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
    #check_env(env)

    model = ppo.PPO.load(os.path.join(currentdir,"./models/PPO/orientation_MP/"+MP_name+"21/70"))
    # model = ppo.PPO.load(os.path.join(currentdir,"./best_models/PPO/orientation_MP/"+MP_name+"18"))
    obs,info = env.reset()
    done = False
    total_rew=0
    displacement, yaw_change = [obs[0]], [obs[1]]
    rew = []
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done,truncated, info = env.step(action)
        displacement.append(obs[0])
        yaw_change.append(obs[1])
        env.render(mode='human')
        total_rew+=reward
        print("act: ",action, reward)
        rew.append(reward)
        if done:
            obs,info = env.reset()
            # print("done: ", rew)
            # total_rew=0
            break
    env.close()
    print(rew)
    print("reward",total_rew)
    plt.subplot(221)
    plt.plot(displacement)
    plt.title("displacement")
    plt.grid()
    plt.subplot(222)
    plt.plot(yaw_change)
    plt.title("yaw_change")
    plt.grid()
    plt.subplot(223)
    plt.plot(rew)
    plt.title("reward")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
