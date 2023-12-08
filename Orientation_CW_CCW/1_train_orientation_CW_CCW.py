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
    model = ppo.PPO("MlpPolicy", env, verbose=1)#,tensorboard_log="./tensorboard/PPO/orientation_MP_CCW/")
    best = 0
    model.learn(total_timesteps=4000)
    model.save("./models/PPO/orientation_MP/"+MP_name+"4")

    # for i in range(20):
    #     print("iteration: ",i)
    #     model.learn(total_timesteps=20)
    #     temp = evaluate_policy(model,model.env,n_eval_episodes=5)
    #     if (temp[0]>best):
    #         model.save("./best_models/PPO/orientation_MP/"+MP_name)
    #         best=temp[0]
    #     if (i%9==0):
    #         model.save("./models/PPO/orientation_MP/"+MP_name)
    # model.save("./models/PPO/orientation_MP/"+MP_name)
if __name__ == '__main__':
    main()
