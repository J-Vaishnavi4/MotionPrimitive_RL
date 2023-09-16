import gym
import numpy.linalg as alg
import numpy as np
from typing import Optional
from gym.spaces import Box

class turtlebot_Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.timestep = 0
        self.gamma_0 = 1.0
        self.gamma_infty = 0.07
        self.rho_max = 0.1
        self.env.observation_space['observation'] = Box(low=-float('inf'), high = float('inf'), shape = (11,))
        
    
    def step(self, action):
        self.timestep+=1
        obs, reward, terminated, truncated, info = self.env.step(action)
        current = obs['achieved_goal']
        goal1 = np.array([1.5,0.43,0.47])
        goal2 = np.array([1.5,1.05,0.47])
        obs['observation'] = np.append(obs['observation'], (self.timestep-100)/50)
        
        
        if(self.timestep<=100):
            robustness = self.rho_max - alg.norm(goal1-current)
            funnel = self.rho_max - (self.gamma_0 - self.gamma_infty)*np.exp(-np.log((self.gamma_0-self.gamma_infty)/(self.rho_max-self.gamma_infty))*(self.timestep/50)) - self.gamma_infty
            reward = (robustness - funnel)/(self.rho_max-funnel) 

        elif(self.timestep>100 and self.timestep<=200):
            robustness = self.rho_max - alg.norm(goal2-current)
            funnel = self.rho_max - (self.gamma_0 - self.gamma_infty)*np.exp(-np.log((self.gamma_0-self.gamma_infty)/(self.rho_max-self.gamma_infty))*((self.timestep-100)/(150-100))) - self.gamma_infty
            reward = (robustness - funnel)/(self.rho_max-funnel) 
        
        info['robustness'] = robustness
        info['funnel'] = funnel
        
        return obs, reward, terminated, truncated, info
        
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.timestep = 0
        obs, info = self.env.reset()
        obs['observation'] = np.append(obs['observation'], (self.timestep-100)/50)
        return obs, info
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()