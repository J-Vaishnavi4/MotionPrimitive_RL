#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv import turtlebot3_burger_GymEnv
import datetime
from stable_baselines3 import ppo
from stable_baselines3.common.env_checker import check_env

def main():

  env = turtlebot3_burger_GymEnv(renders=True, isDiscrete=False)
# It will check your custom environment and output additional warnings if needed
  #check_env(env)

  # model = ppo.PPO("MlpPolicy", env, verbose=1)
  print("############Training completed################")
  model = ppo.PPO.load(os.path.join(currentdir,"burger_hrading_goal"))

  obs = env.reset()
  done = False
  rew=0
#   while not done:
  for i in range(10000):
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
    #   print("reward is ", reward)
      env.render(mode='human')
      rew+=reward
      if done:
        obs = env.reset()
        print("done: ", rew)
        rew=0

  env.close()

if __name__ == '__main__':
  main()