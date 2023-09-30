import gymnasium as gym
import numpy as np
import math
import pybullet as p
# from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt
# from simple_driving.resources.turtlebot_env import TurtleBot
import time
import random

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(                                         #action_space : linear Vel, ang vel (normalized)
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(                                    #observation space: robot_x, robot_y, cos(yaw), sin(yaw), vel_x, vel_y, goal_x, goal_y
            low=np.array([-10.0, -10.0, -1.0, -1.0, -5.0, -5.0, -10.0, -10.0]),
            high=np.array([10.0, 10.0, 1.0, 1.0, 5.0, 5.0, 10.0, 10.0]), dtype=np.float64)
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.GUI)
        
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)
        self.observation  = np.array([0,0,0,0])
        self.turtlebot = None
        self.goal = None
        # from turtlebot_env.py----------------------------------------
        # p.setGravity(0,0,-10)
        # offset = [0,0,0.1]
        # self.turtle = p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/MotionPrimitive_RL/turtlebot3_description/urdf/Edit_turtlebot3_burger.urdf.xacro',offset)
        # self.plane = p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/MotionPrimitive_RL/simple_driving/resources/simpleplane.urdf')
        # p.setRealTimeSimulation(1)
        # self.wheel_joints = [1, 2]
        # # Joint speed
        # self.joint_speed = 0

        #--------------------------------------------------------------
        
        self.done = False
        self.prev_dist_to_goal = None
        # self.prev_orientation = None
        self.initial_orientation = None
        self.prev_velocity = None
        self.prev_robot_goal_relative_pos = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

    def get_observations(self):
        pos, ang = p.getBasePositionAndOrientation(self.turtle, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]                                                       # Position- X, Y
        # Get the velocity of the car
        vel = p.getBaseVelocity(self.turtle, self.client)[0][0:2]           # Velocities- Vx, Vy
        # Concatenate position, orientation, velocity
        car_ob = (pos + ori + vel)
        return car_ob
    def step(self, action):
        # Feed action to the car and get observation of car's state

        lin_vel,ang_vel = 4*action[0],4*action[1]
        L = 0.16
        R = 0.033

        # Clip throttle and steering angle to reasonable values
        lin_vel = min(max(lin_vel, 0), 1)
        ang_vel = max(min(ang_vel, 0.6), -0.6)

        rightWheelVelocity = (2*lin_vel + ang_vel*L)/(2*R)
        leftWheelVelocity = (2*lin_vel - ang_vel*L)/(2*R)

        # print("action: ",leftWheelVelocity,rightWheelVelocity)
        p.setJointMotorControlArray(self.turtle,[1,2],p.VELOCITY_CONTROL, targetVelocities=[leftWheelVelocity,rightWheelVelocity],forces=[10,10])

        p.stepSimulation()
        # Get the position and orientation of the car in the simulation
        car_ob = self.get_observations()
        
        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        currect_orientation = np.arctan(car_ob[2]/car_ob[3])
        current_velocity = math.sqrt(((car_ob[4])**2)+(car_ob[5])**2)
        current_robot_goal_relative_pos = tuple(map(lambda i, j: i - j, self.goal, car_ob[0:2])) # self.goal - car_ob[0:2]

        """ REWARDS DURING THE EPISODE
         rew1: positive reward if the robot moves towards goal
         rew2: penalty if the robot moves away from goal
         rew3: penalty if the orientation varies from the robot's initial orientation
         rew4: positive reward if robot reduces its velocity once it is close to goal (20cm), penalty if velocity increases
         rew5: need to define to avoid overshoot situation: dot product of relative position vectors of goal and turtlebot in current and previous step
                would be negative (= -1) if overshoot happens in the current step, BUT THIS NEEDS POSITION VECTORS OF GOAL AND ROBOT
        """
        # Done by running off boundaries
        if (car_ob[0] >= 9 or car_ob[0] <= -9 or
            car_ob[1] >= 9 or car_ob[1] <= -9):
            reward = -50
            print("out of plane")
            self.done = True
        # Done by reaching goal
        elif dist_to_goal < 0.05:       #Should we add maximum episode length as termination criteria?
            self.done = True
            print("successful")
            reward = 50
        # Rewards during the episode
        else:

            rew1 = (self.prev_dist_to_goal - dist_to_goal)*(dist_to_goal-self.prev_dist_to_goal<0)
            rew2 = -(dist_to_goal-self.prev_dist_to_goal)*(dist_to_goal-self.prev_dist_to_goal>0)
            rew3 = -(np.absolute(self.initial_orientation - currect_orientation))
            rew4 = 1*(dist_to_goal < 0.2 and current_velocity < self.prev_velocity) + (-1)*(dist_to_goal < 0.2 and current_velocity > self.prev_velocity)
            # if (dist_to_goal<0.1):
                # rew4 = skewed_gaussian_reward?
            rew5 = 1*(sum(tuple(ele1*ele2 for ele1,ele2 in zip(current_robot_goal_relative_pos,self.prev_robot_goal_relative_pos))))
            reward = rew1 + rew2 + rew3 + rew4 + rew5
            # print("during episode",dist_to_goal, reward)
        self.prev_dist_to_goal = dist_to_goal
        # self.prev_orientation = currect_orientation
        self.prev_velocity = current_velocity
        self.prev_robot_goal_relative_pos = current_robot_goal_relative_pos
        # states = car_ob[2:], dist_to_goal,
        states = np.array((car_ob + self.goal))# + tuple([time1-self.start_time]))         #need to keep states as dist_to_goal, velocities and time(?)
        truncated = self.done
        info = {}
        return states, reward, self.done, truncated, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane and car
        # Plane(self.client)
        offset = [0,0,0.5]
        self.turtle = p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/MotionPrimitive_RL/turtlebot3_description/urdf/Edit_turtlebot3_burger.urdf.xacro',offset)
        self.plane = p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/MotionPrimitive_RL/simple_driving/resources/simpleplane.urdf')
        p.setRealTimeSimulation(1)
        self.wheel_joints = [1, 2]
        # Joint speed
        self.joint_speed = 0
        # self.turtlebot = TurtleBot(self.client)

        # Set the goal at a distance "des_dist" along its orientation
        # des_dist = 1
        des_dist = random.uniform(1,10)                                          # Generates random number between 1 and 10 - for desired distance
        # Get observation to return
        car_ob = self.get_observations()
        x = car_ob[0] + des_dist*car_ob[2]
        y = car_ob[1] + des_dist*car_ob[3]
        self.goal = (x, y)
        self.done = False
        # Visual element of the goal
        # Goal(self.client, self.goal)

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                           (car_ob[1] - self.goal[1]) ** 2))
        # self.prev_orientation = np.arctan(car_ob[3]/car_ob[2])
        self.initial_orientation = np.arctan(car_ob[3]/car_ob[2])
        self.prev_velocity = math.sqrt(((car_ob[4])**2)+(car_ob[5])**2)
        self.prev_robot_goal_relative_pos = tuple(map(lambda i, j: i - j, self.goal, car_ob[0:2])) #self.goal - car_ob[0:2]
        info = {}
        return np.array((car_ob + self.goal)), info           # dictionary to keep additional info as per stable_baselines3
        # return np.array((car_ob + self.goal)) # + tuple([time1 - self.start_time]))

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
