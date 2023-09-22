import gym
import torch
from turtlebot_agent import TRPOAgent
import simple_driving
import time
import pybullet as p
from simple_driving.envs import turtlebot_simple_driving_env

def main():
    nn = torch.nn.Sequential(torch.nn.Linear(9, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 2))
    agent = TRPOAgent(policy=nn)

    agent.train("turtlebot_env", seed=0, batch_size=5000, iterations=10,
                max_episode_length=250, verbose=True)
    agent.save_model("agent_turtlebot.pth")
    agent.load_model("agent_turtlebot.pth")

    # env = gym.make('SimpleDriving-v0')
    env = turtlebot_simple_driving_env.SimpleDrivingEnv()
    ob = env.reset()
    reward = 0
    done= False
    while not done: #True:
        action = agent(ob)
        ob, reward, done, _ = env.step(action)
        reward += reward
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1/30)
    print("Final reward at the end of the episode: ", reward)

if __name__ == '__main__':
    main()
