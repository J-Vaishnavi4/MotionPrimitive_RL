from stable_baselines3 import TD3, PPO, HerReplayBuffer, TRPO
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import numpy as np
from Wrapper import Wrapper

model_class = TRPO
env = Wrapper(gym.make("turtlebot-v0", max_episode_steps = 500))

model = model_class("MultiInputPolicy", env, verbose=0)

best = -100
plot_reward = []
for i in range(400):
    model.learn(total_timesteps = 5000)
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