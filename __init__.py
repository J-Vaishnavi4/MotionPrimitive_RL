from simple_driving.envs.simple_driving_env import SimpleDrivingEnv
from gymnasium.envs.registration import register
register(
        id="TurtlebotEnv-v0",
        entry_point="sb3_turtlebot_simple_driving_env:SimpleDrivingEnv",
    )