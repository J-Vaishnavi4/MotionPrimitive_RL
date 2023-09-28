import pybullet as p
import os
import math
import gym
import time

class TurtleBot:
    def __init__(self, client):
        self.client = client
        p.connect(p.DIRECT)
        p.setGravity(0,0,-10)
        offset = [0,0,0]
        self.turtle = p.loadURDF('/home/vaishnavi/Documents/IISc/RL/Coding/MotionPrimitive_RL_1/turtlebot3_description/urdf/Edit_turtlebot3_burger.urdf.xacro',offset)
        self.plane = p.loadURDF('/home/vaishnavi/Documents/IISc/RL/Coding/MotionPrimitive_RL_1/simple_driving/resources/simpleplane.urdf')
        p.setRealTimeSimulation(1)
        self.wheel_joints = [1, 2]
        # Joint speed
        self.joint_speed = 0

    def get_ids(self):
        return self.turtle, self.client

    def apply_action(self, action):
        # Expects action to be two dimensional
        leftWheelVelocity,rightWheelVelocity = 1,1#action[0],action[1]
        # print("action: ",leftWheelVelocity,rightWheelVelocity)
        p.setJointMotorControlArray(self.turtle,[1,2],p.VELOCITY_CONTROL, targetVelocities=[leftWheelVelocity,rightWheelVelocity],forces=[10,10])
        # p.stepSimulation()
    def get_observation(self):
        # Get the position and orientation of the car in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.turtle, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]                                                       # Position- X, Y
        # Get the velocity of the car
        vel = p.getBaseVelocity(self.turtle, self.client)[0][0:2]           # Velocities- Vx, Vy
        # Concatenate position, orientation, velocity
        observation = (pos + ori + vel)
        # print("obs: ", observation)

        return observation
