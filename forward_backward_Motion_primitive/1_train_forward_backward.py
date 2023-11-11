#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv_forward import turtlebot3_burger_GymEnv_forward
from turtlebot3_burger_GymEnv_backward import turtlebot3_burger_GymEnv_backward

import datetime
from stable_baselines3 import ppo
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

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
    check_env(env)
    model = ppo.PPO("MlpPolicy", env, verbose=0)#,tensorboard_log="./tensorboard/PPO/forward_MP_4/")
    best = 0
    for i in range(100):
        print("iteration: ",i)
        model.learn(total_timesteps=1000)
        temp = evaluate_policy(model,model.env,n_eval_episodes=5)
        if (temp[0]>best):
            model.save("./best_models/PPO/translation_MP"+MP_name)
            best=temp[0]
        if (i%10==0):
            model.save("./models/PPO/translation_MP"+MP_name)
        model.save("./models/PPO/translation_MP"+MP_name)
if __name__ == '__main__':
    main()
