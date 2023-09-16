import gymnasium as gym
import numpy as np
import math
import pybullet as p
# from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt
from simple_driving.resources.turtlebot_env import TurtleBot
import time 

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-10, -10], dtype=np.float32),
            high=np.array([10, 10], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-10, -10, -1, -1, -5, -5, -10, -10, 0], dtype=np.float32),
            high=np.array([10, 10, 1, 1, 5, 5, 10, 10, math.inf], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)
        self.observation  = np.array([0,0,0,0])
        self.turtlebot = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.prev_orientation = None
        self.prev_velocity = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.start_time = time.time()
        self.reset()

    def step(self, action):
        # Feed action to the car and get observation of car's state
        self.turtlebot.apply_action(action)
        p.stepSimulation()
        car_ob = self.turtlebot.get_observation()
        time1 = time.time()

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        currect_orientation = np.arctan(car_ob[2]/car_ob[3])
        current_velocity = math.sqrt(((car_ob[4])**2)+(car_ob[5])**2)

        """ REWARDS DURING THE EPISODE
         rew1: positive reward if the robot moves towards goal 
         rew2: penalty if the robot moves away from goal 
         rew3: penalty if the orientation is changed (this code is for straight line motion) but need to check this,
                because we can't penalize if the orientation is being corrected
                Maybe need to save the robot's original orientation to bring it back to the original orientation
         rew4: positive reward if robot reduces its velocity once it is close to goal, penalty if velocity increases
         rew5: need to define to avoid overshoot situation
        """
        # Done by running off boundaries
        if (car_ob[0] >= 10 or car_ob[0] <= -10 or
            car_ob[1] >= 10 or car_ob[1] <= -10):
            reward = -50
            self.done = True
        # Done by reaching goal
        elif dist_to_goal < 0.05:
            self.done = True
            reward = 50
        # Rewards during the episode
        else:
            rew1 = (self.prev_dist_to_goal - dist_to_goal)*(dist_to_goal-self.prev_dist_to_goal<0)
            rew2 = -(dist_to_goal-self.prev_dist_to_goal)*(dist_to_goal-self.prev_dist_to_goal>0)
            rew3 = -(np.absolute(self.prev_orientation - currect_orientation))  
            rew4 = 1*(dist_to_goal < 0.2 and current_velocity < self.prev_velocity) + (-1)*(dist_to_goal < 0.2 and current_velocity > self.prev_velocity)
            reward = rew1 + rew2 + rew3 + rew4
        self.prev_dist_to_goal = dist_to_goal
        self.prev_orientation = currect_orientation
        self.prev_velocity = current_velocity
        states = car_ob[2:], dist_to_goal,
        self.start_time = time1
        ob = np.array((car_ob + self.goal) + tuple([time1-self.start_time]), dtype=np.float32)         #need to keep states as dist_to_goal, velocities and time(?)
        truncated = False
        return ob, reward, self.done, truncated, {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.start_time = time.time() 
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self.client)
        self.turtlebot = TurtleBot(self.client)

        # Set the goal at a distance "des_dist" along its orientation
        des_dist = 1
        # Get observation to return
        car_ob = self.turtlebot.get_observation()
        time1 = time.time()
        x = car_ob[0] + des_dist*car_ob[2]
        y = car_ob[1] + des_dist*car_ob[3]
        self.goal = (x, y)
        self.done = False
        # self.timestep = 0
        # Visual element of the goal
        Goal(self.client, self.goal)

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                           (car_ob[1] - self.goal[1]) ** 2))
        self.prev_orientation = np.arctan(car_ob[3]/car_ob[2])
        self.prev_velocity = math.sqrt(((car_ob[4])**2)+(car_ob[5])**2)
        return np.array((car_ob + self.goal)+ tuple([time1 - self.start_time]), dtype=np.float32), {}           # dictionary to keep additional info as per stable_baselines3 
    
    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.turtlebot.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)
