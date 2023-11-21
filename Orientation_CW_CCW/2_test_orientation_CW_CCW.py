#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
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
    #check_env(env)

    model = ppo.PPO.load(os.path.join(currentdir,"./best_models/PPO/orientation_MP/"+MP_name))

    obs,info = env.reset()
    done = False
    rew=0

    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done,truncated, info = env.step(action)
        env.render(mode='human')
        rew+=reward
        if done:
            obs,info = env.reset()
            print("done: ", rew)
            rew=0

        env.close()

if __name__ == '__main__':
    main()
