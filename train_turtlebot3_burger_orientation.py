#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv_orientation import turtlebot3_burger_GymEnv_orientation
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def main():
  env = turtlebot3_burger_GymEnv_orientation(renders=True, isDiscrete=False)
# It will check your custom environment and output additional warnings if needed
  # check_env(env)

  model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./tensorboard/orientation_PPO/")
  model.learn(100000)
  print("############Training completed################")
  model.save(os.path.join(currentdir,"burger_orientation"))
  obs = env.reset()
  env.close()

if __name__ == '__main__':
  main()
