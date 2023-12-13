import os
currentdir = os.path.dirname(__file__)
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
        euler_offset = (0, 0, 0)
        quaternion_offset = self._p.getQuaternionFromEuler(euler_offset)
        robot = self._p.loadURDF(currentdir+'/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro',[0,0,0],baseOrientation=quaternion_offset)

        self.robotUniqueId = robot

        self.Wheels = [1,2] #1 is left 2 is right
        self.maxForce = 350
        self.nMotors = 2
        self.speedMultiplierRight = 1*1
        self.speedMultiplierLeft = 1*1

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
        observation.append(list(pos))
        observation.append(self._p.getEulerFromQuaternion(orn)[2])
        observation.append(np.linalg.norm(linV))
        observation.append(np.linalg.norm(angV))
        return observation

    def applyAction(self, lin_vel,ang_vel):
        L = 0.160
        R = 0.033
        lin_vel = 0.22*lin_vel
        ang_vel = 2.84*ang_vel
        targetVelocityRight = (2*lin_vel + ang_vel*L)/(2*R) * self.speedMultiplierLeft
        targetVelocityLeft = (2*lin_vel - ang_vel*L)/(2*R) * self.speedMultiplierLeft
        # print("left and right wheel velocities: ", targetVelocityLeft, targetVelocityRight, targetVelocityLeft-targetVelocityRight)
        self._p.setJointMotorControlArray(self.robotUniqueId,[1,2],self._p.VELOCITY_CONTROL, targetVelocities=[targetVelocityLeft,targetVelocityRight],forces=[self.maxForce,self.maxForce])
