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
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    MP_name = input("Clockwise(CW)/ Counter Clockwise (CCW) MP: ")
    if MP_name == "CW":
        env = turtlebot3_burger_GymEnv_CW(renders=True, isDiscrete=False)
    elif MP_name == "CCW":
        env = turtlebot3_burger_GymEnv_CCW(renders=True, isDiscrete=False)
    else:
        raise SystemExit("Incorrect MP name")

    check_env(env)
    model = ppo.PPO("MlpPolicy", env, verbose=1,tensorboard_log="./tensorboard/PPO/orientation_MP_CCW/"+MP_name+"21")

    best = 0.01
    best_yaw = 0
    v = 1000
    for i in range(100):
        print("iteration: ",i)
        v = v+200
        model.learn(total_timesteps=v, reset_num_timesteps = False)
        done = False
        obs,info = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done,truncated, info = env.step(action)
        if (obs[0] < best):# and obs[1]>best_yaw):
            model.save("./best_models/PPO/orientation_MP/"+MP_name+"21")
            best = obs[0]
            best_yaw = obs[1]
        if (i%10==0):
            model.save("./models/PPO/orientation_MP/"+MP_name+"21/"+str(i))
    model.save("./models/PPO/orientation_MP/"+MP_name+"21/"+str(i))

if __name__ == '__main__':
    main()
