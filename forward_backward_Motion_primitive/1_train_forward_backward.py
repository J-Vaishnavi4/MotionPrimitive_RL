#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv_forward import turtlebot3_burger_GymEnv_forward
from turtlebot3_burger_GymEnv_backward import turtlebot3_burger_GymEnv_backward
# from turtlebot3_burger_GymEnv_forward_4states import turtlebot3_burger_GymEnv_forward

from stable_baselines3 import ppo, SAC

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
    model = ppo.PPO("MlpPolicy", env, verbose=1, n_steps=4000,tensorboard_log="./tensorboard/PPO/translation_MP/"+MP_name+"14")

    # for i in range(500):
    #     model.learn(total_timesteps=20000, reset_num_timesteps = False)
    #     model.save(("./models/PPO/translation_MP/"+MP_name+"8/"+str(i)))
    
    best_dist = 0.01
    best_ld = 1
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
        if (obs[0] > best_dist and info['ld'] < best_ld):
            model.save("./best_models/PPO/translation_MP/"+MP_name+"14")
            best_dist = obs[0]
            best_ld = info['ld']
        if (i%10==0):
            model.save("./models/PPO/translation_MP/"+MP_name+"14/"+str(i))
    model.save("./models/PPO/translation_MP/"+MP_name+"14/"+str(i))

if __name__ == '__main__':
    main()
