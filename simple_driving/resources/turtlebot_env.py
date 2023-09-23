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
        self.turtle = p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/MotionPrimitive_RL/turtlebot3_description/urdf/Edit_turtlebot3_burger.urdf.xacro',offset)
        self.plane = p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/MotionPrimitive_RL/simple_driving/resources/simpleplane.urdf')
        self.wheel_joints = [1, 2]
        # Joint speed
        self.joint_speed = 0

    def get_ids(self):
        return self.turtle, self.client

    def apply_action(self, action):
        # Expects action to be two dimensional
        leftWheelVelocity,rightWheelVelocity = action[0],action[1]
        p.setJointMotorControlArray(self.turtle,[1,2],p.VELOCITY_CONTROL, targetVelocities=[leftWheelVelocity,rightWheelVelocity],forces=[1000,1000])

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
