import os
import copy
import math

import numpy as np

class TurtleBot3:

    def __init__(self, bullet_client, urdfRootPath = '', timeStep = 0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self._p = bullet_client
        self.reset()

    def reset(self):
        robot = self._p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/MotionPrimitive_RL/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro',[0,0,0])
        #plane = self._p.loadURDF("plane_transparent.urdf")
        # plane = self._p.loadURDF('/home/tushar-20-msi/turtlebot_stable_baseline/turtlebot3_description/urdf/simpleplane.urdf')

        self.robotUniqueId = robot

        self.Wheels = [1,2] #1 is left 2 is right
        self.maxForce = 350
        self.nMotors = 2 
        self.speedMultiplierRight = 0.5
        self.speedMultiplierLeft = 0.5

        #no. of actions
        self.no_of_actions = 2

    def getActionDimension(self):
        return self.no_of_actions

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        pos, orn = self._p.getBasePositionAndOrientation(self.robotUniqueId)
        linV, angV = self._p.getBaseVelocity(self.robotUniqueId)

        observation.extend(list(pos))
        observation.extend(list(self._p.getEulerFromQuaternion(orn)))
        observation.extend(list(linV))
        observation.extend(list(angV))

        return observation

    def applyAction(self, lin_vel,ang_vel):
        L = 0.16
        R = 0.033
        targetVelocityRight = (2*lin_vel + ang_vel*L)/(2*R) * self.speedMultiplierLeft
        targetVelocityLeft = (2*lin_vel - ang_vel*L)/(2*R) * self.speedMultiplierLeft

        self._p.setJointMotorControlArray(self.robotUniqueId,[1,2],self._p.VELOCITY_CONTROL, targetVelocities=[targetVelocityLeft,targetVelocityRight],forces=[self.maxForce,self.maxForce])

