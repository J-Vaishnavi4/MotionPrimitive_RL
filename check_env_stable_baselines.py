from stable_baselines3.common.env_checker import check_env
from simple_driving.envs import turtlebot_simple_driving_env

env = turtlebot_simple_driving_env.SimpleDrivingEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)