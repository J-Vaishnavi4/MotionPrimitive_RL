#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv_orientation import turtlebot3_burger_GymEnv_orientation
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
 
def main():
  env = turtlebot3_burger_GymEnv_orientation(renders=True, isDiscrete=False)
# It will check your custom environment and output additional warnings if needed
  # check_env(env)

  model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./tensorboard/PPO/orientation_MP_360degree/")
  best=0
  for i in range(100):
      model.learn(total_timesteps=10000)
      temp = evaluate_policy(model,model.env,n_eval_episodes=5)
      if (temp[0]>best):
          model.save("./best_models/PPO/orientation_MP_360degree")
          best=temp[0]
      if (i%10==0):
          model.save("./models/PPO/orientation_MP_360degree")

  # model.learn(100000)
  # print("############Training completed################")
  # model.save(os.path.join(currentdir,"burger_orientation_360degree"))
  # obs = env.reset()
  # env.close()

if __name__ == '__main__':
  main()