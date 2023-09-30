import gymnasium as gym
import torch
from turtlebot_agent import TRPOAgent
import simple_driving
import time
import pybullet as p
from simple_driving.envs import turtlebot_simple_driving_env
from gymnasium.envs.registration import register

def main():
    # register(
    #     id="TurtlebotEnv-v0",
    #     entry_point="simple_driving.envs.turtlebot_simple_driving_env:SimpleDrivingEnv",
    # )
    # env = gym.make("TurtlebotEnv-v0")
    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 2))
    agent = TRPOAgent(policy=nn)

    agent.train("TurtlebotEnv-v0", seed=0, batch_size=5000, iterations=5,
                max_episode_length=10000, verbose=True)
    agent.save_model("agent_turtlebot.pth")
    agent.load_model("agent_turtlebot.pth")

    # env = gym.make('SimpleDriving-v0')
    print("after train")
    env = turtlebot_simple_driving_env.SimpleDrivingEnv()
    ob = env.reset()
    reward = 0
    done= False
    # print("test")
    # while not done: #True:
    #     action = agent(ob)
    #     ob, reward, done, _ = env.step(action)
    #     reward += reward
    #     env.render()
    #     if done:
    #         ob = env.reset()
    #         time.sleep(1/30)
    # print("Final reward at the end of the episode: ", reward)

if __name__ == '__main__':
    main()
