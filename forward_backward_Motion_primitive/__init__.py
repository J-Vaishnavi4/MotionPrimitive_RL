from gymnasium.envs.registration import register
from turtlebot3_burger_GymEnv_forward import turtlebot3_burger_GymEnv_forward
from turtlebot3_burger_GymEnv_backward import turtlebot3_burger_GymEnv_backward

register(
    # unique identifier for the env `name-version`
    id="Turtlebot_waffle-F-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="turtlebot3_burger_GymEnv_forward:turtlebot3_burger_GymEnv_forward",
)

register(
    # unique identifier for the env `name-version`
    id="Turtlebot_waffle-B-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="turtlebot3_burger_GymEnv_backward:turtlebot3_burger_GymEnv_backward",
)