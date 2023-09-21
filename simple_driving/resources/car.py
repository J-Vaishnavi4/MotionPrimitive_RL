import pybullet as p
import os
import math


class Car:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), '/home/tushar-20-msi/pblt_learning/turtlebot_burger/turtlebot3-master/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro')
        self.car = p.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0.1],
                              physicsClientId=client)

        # Joint indices as found by p.getJointInfo()
        # self.steering_joints = [0, 2]
        # self.drive_joints = [1, 3, 4, 5]
        self.joints = [1,2]
        # Joint speed
        self.joint_speed = 0
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant increases "speed" of the car
        self.c_throttle = 20

    def get_ids(self):
        return self.car, self.client

    def apply_action(self, action):
        # Expects action to be two dimensional
        throttle, steering_angle = action
        L = 0.16
        R = 0.033

        # Clip throttle and steering angle to reasonable values
        throttle = min(max(throttle, 0), 1)
        steering_angle = max(min(steering_angle, 0.6), -0.6)

        rightWheelVelocity = (2*throttle + steering_angle*L)/(2*R)
        leftWheelVelocity = (2*throttle - steering_angle*L)/(2*R)


        # Set the steering joint positions
        # p.setJointMotorControlArray(self.car, self.steering_joints,
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPositions=[steering_angle] * 2,
        #                             physicsClientId=self.client)

        # Calculate drag / mechanical resistance ourselves
        # Using velocity control, as torque control requires precise models
        friction = -self.joint_speed * (self.joint_speed * self.c_drag +
                                        self.c_rolling)
        acceleration = self.c_throttle * throttle + friction
        # Each time step is 1/240 of a second
        self.joint_speed = self.joint_speed + 1/30 * acceleration
        if self.joint_speed < 0:
            self.joint_speed = 0

        # Set the velocity of the wheel joints directly
        # p.setJointMotorControlArray(
        #     bodyUniqueId=self.car,
        #     jointIndices=self.drive_joints,
        #     controlMode=p.VELOCITY_CONTROL,
        #     targetVelocities=[self.joint_speed] * 4,
        #     forces=[1.2] * 4,
        #     physicsClientId=self.client)

        p.setJointMotorControl2(bodyIndex=self.car,jointIndex=1,controlMode=p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000,physicsClientId=self.client)
        p.setJointMotorControl2(bodyIndex=self.car,jointIndex=2,controlMode=p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000,physicsClientId=self.client)

    def get_observation(self):
        # Get the position and orientation of the car in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.car, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]
        # Get the velocity of the car
        vel = p.getBaseVelocity(self.car, self.client)[0][0:2]

        # Concatenate position, orientation, velocity
        observation = (pos + ori + vel)

        return observation








