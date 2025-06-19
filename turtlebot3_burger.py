import os
currentdir = os.path.dirname(__file__)
import copy
import math
import numpy as np
import csv

class TurtleBot3:

    def __init__(self, bullet_client, urdfRootPath = '', timeStep = 0.01, init_angle=np.radians(2), case = "1", init_position = [0,0]):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.init_angle = init_angle
        self._p = bullet_client
        self._case = case
        self.init_position = init_position
        self.reset()

    def reset(self):
        pts = np.arange(-180,181,1)
        init_angle = np.random.choice(np.radians(pts))
        # self.init_angle = init_angle
        # print("angle: ", np.rad2deg(self.init_angle))
        euler_offset = (0, 0, 0)
        quaternion_offset = self._p.getQuaternionFromEuler(euler_offset)
        init_x = 0#self.init_position[0]
        init_y = 0#self.init_position[1]
        # print("init : ", self.init_position)
        if self._case == "1" or self._case == "3":
            robot = self._p.loadURDF(currentdir+'/turtlebot3_description/urdf/tm_turtlebot3.urdf',[init_x,init_y,0],baseOrientation=quaternion_offset)
        else:
            robot = self._p.loadURDF(currentdir+'/turtlebot3_description/urdf/tm_turtlebot3_diff_radius.urdf',[init_x,init_y,0],baseOrientation=quaternion_offset)
        self.robotUniqueId = robot

        self.Wheels = [1,2] #1 is left 2 is right
        self.maxForce = 350
        self.nMotors = 2
        self.speedMultiplierRight = 1
        self.speedMultiplierLeft = 1

        #no. of actions
        self.no_of_actions = 2

    def getActionDimension(self):
        return self.no_of_actions

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        pos, orn = self._p.getBasePositionAndOrientation(self.robotUniqueId)
        linVw, angVw = self._p.getBaseVelocity(self.robotUniqueId)
        rotn_m =np.array(self._p.getMatrixFromQuaternion(orn)).reshape(3,3)
        linVb = np.dot(rotn_m.T,np.array(linVw))
        observation.append(list(pos))
        observation.append(self._p.getEulerFromQuaternion(orn)[2])
        observation.append(linVb[0])
        observation.append(linVb[1])
        return observation

    def applyAction(self, left_vel, right_vel):
        if self._case == "1" or self._case == "2":
            targetVelocityRight = 5.96*right_vel
            targetVelocityLeft = 5.96*left_vel
        if self._case == "3":
            targetVelocityRight = 5.96*right_vel
            targetVelocityLeft = 3.0*left_vel
        self._p.resetBaseVelocity(self.robotUniqueId,[0.1,0,0], angularVelocity=[0,0,0])
        self._p.setJointMotorControlArray(self.robotUniqueId,[1,2],self._p.VELOCITY_CONTROL, targetVelocities=[targetVelocityLeft,targetVelocityRight],forces=[self.maxForce,self.maxForce])