from stable_baselines3 import TD3, PPO, HerReplayBuffer
from sb3_contrib import TRPO
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
import sb3_turtlebot_simple_driving_env

register(
        id="TurtlebotEnv-v0",
        entry_point="sb3_turtlebot_simple_driving_env:SimpleDrivingEnv",
    )
env = gym.make("TurtlebotEnv-v0")
# env = sb3_turtlebot_simple_driving_env.SimpleDrivingEnv()
model_class = TRPO

model = model_class("MlpPolicy", env, verbose=1)

best = -100
plot_reward = []
for i in range(20):
    model.learn(total_timesteps = 100)
    temp = evaluate_policy(model, model.env, n_eval_episodes = 5)
    print("reward: ", temp,flush=True)
    plot_reward.append(temp)
    if(temp[0]>best):
        model.save("./best_model/turtlebot_")
        best = temp[0]
    if(i%10==0):
        model.save("./model/turtlebot_")

np.save('turtlebot.npy', np.array(plot_reward))
model.save("./model/turtlebot_")