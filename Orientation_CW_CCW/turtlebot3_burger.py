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
        init_angle =np.random.uniform(low=-math.pi, high=math.pi)
        init_x =np.random.uniform(low=-10, high=10)
        init_y =np.random.uniform(low=-10, high=10)
        # init_angle, init_x, init_y = -1.138214407202498, -5.726495713586985, -51.814113092004945    #CCW Sample
        # init_angle, init_x, init_y = -0.809946694764633, 17.196942333320962, 51.16743580960241      #CW Sample
        # init_angle, init_x, init_y = 1.483170738523647, 6.725357855027365, -0.7632824475452686     #CW_24
        # init_angle, init_x, init_y = -2.097916968913103, 3.658051386056279, -1.9265580881536781      #CCW_24
        euler_offset = (0, 0, init_angle)
        quaternion_offset = self._p.getQuaternionFromEuler(euler_offset)
        # robot = self._p.loadURDF(currentdir+'/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro',[0,0,0],baseOrientation=quaternion_offset)
        
        robot = self._p.loadURDF(currentdir+'/turtlebot3_description/urdf/tm_turtlebot3.urdf',[init_x,init_y,0],baseOrientation=quaternion_offset)

        # print(init_angle, init_x, init_y)
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

        self._p.setJointMotorControlArray(self.robotUniqueId,[1,2],self._p.VELOCITY_CONTROL, targetVelocities=[targetVelocityLeft,targetVelocityRight],forces=[self.maxForce,self.maxForce])
