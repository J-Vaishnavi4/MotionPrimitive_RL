#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from Gym_env_translation import translation_env
from Gym_env_rotation import rotation_env

from stable_baselines3 import ppo, SAC

def main():
    
    case = "1" #input("1: Ideal case, 2: Diff wheel radius, 3: Diff max wheel velocity: ")
    algorithm = "1" #input("1: PPO, 2: SAC: ")
    MP_name = input("1: CW, 2: CCW: 3: Forward, 4: Backward: ")
    if MP_name == "1":
        MP_name = "CW"
        env = rotation_env(renders=True, isDiscrete=False, motion="CW", case = case)
    elif MP_name == "2":
        MP_name = "CCW"
        env = rotation_env(renders=True, isDiscrete=False, motion="CCW", case = case)
    elif MP_name == "3":
        MP_name = "Forward"
        env = translation_env(renders=True, isDiscrete=False, motion="Forward", case = case)
    elif MP_name == "4":
        MP_name = "Backward"
        env = translation_env(renders=True, isDiscrete=False, motion="Backward", case = case)
    else:
        raise SystemExit("Incorrect MP name")
    
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
        model = ppo.PPO("MlpPolicy", env, verbose= 1, device = "cpu" , tensorboard_log="./"+case+"/"+MP_name+"/tensorboard/model_0p1")
    elif algorithm == "2":
        algo = "SAC"
        model = SAC("MlpPolicy", env, verbose= 1, device = "cpu" , tensorboard_log="./"+case+"/"+MP_name+"/tensorboard/model_0p1")
    else:
        raise SystemExit("Incorrect algorithm name")
    v = 1000
    for i in range(1000):
        print("iteration: ",i)
        model.learn(total_timesteps=v, reset_num_timesteps = False)
        model.save("./"+case+"/"+MP_name+"/models/"+algo+"/model_0p1/"+str(i))
    # model.save("./"+case+"/"+MP_name+"/models/"+algorithm+"/model_2/"+str(i))


if __name__ == '__main__':
    main()
