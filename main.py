import gym
import torch
from agent import TRPOAgent
import simple_driving
import time
import pybullet as p
from simple_driving.envs import turtlebot_simple_driving_env

def main():
    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 2))
    agent = TRPOAgent(policy=nn)

    agent.load_model("agent.pth")
    agent.train("SimpleDriving-v0", seed=0, batch_size=5000, iterations=10,
                max_episode_length=250, verbose=True)
    agent.save_model("agent.pth")
    print("after training")
    env = gym.make('SimpleDriving-v0')
    # env = turtlebot_simple_driving_env.SimpleDrivingEnv()
    ob = env.reset()
    done= False
    while not done: #True:
        action = agent(ob)
        ob, _, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1/30)


if __name__ == '__main__':
    main()
