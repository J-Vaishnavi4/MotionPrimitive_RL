import pybullet as p
import time
import math
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
p.connect(p.GUI)
p.resetSimulation()
offset = [0,0,0]
init_angle =1*math.pi/4
euler_offset = (0, 0, init_angle)
quaternion_offset = p.getQuaternionFromEuler(euler_offset)
turtle1 = p.loadURDF('/home/vaishnavi/Documents/IISc/RL/Coding/Car-Plane robot_RL/MotionPrimitive_RL/forward_backward_Motion_primitive/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro',offset,baseOrientation=quaternion_offset)
# turtle2 = p.loadURDF('/home/tushar-20-msi/turtlebot_stable_baseline/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro',\
# 					basePosition=[0,0,0], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
plane = p.loadURDF('/home/vaishnavi/Documents/IISc/RL/Coding/Car-Plane robot_RL/MotionPrimitive_RL/forward_backward_Motion_primitive/turtlebot3_description/urdf/simpleplane.urdf')
L = 0.16
R = 0.033
p.setRealTimeSimulation(1)

ang_vel = p.addUserDebugParameter('Angular_vel', -10, 10, 0)
lin_vel = p.addUserDebugParameter('linear_vel', -2, 2, 0)

position, orientation = p.getBasePositionAndOrientation(turtle1)

for j in range (p.getNumJoints(turtle1)):
	print(p.getJointInfo(turtle1,j))
forward=0
turn=0
leftWheelVelocity=0
rightWheelVelocity=0
for i in range (40):
	p.setGravity(0, 0, -10)
	time.sleep(1./10)
	user_ang_vel = 0
	user_lin_vel = 0.22

	rightWheelVelocity = (2*user_lin_vel + user_ang_vel*L)/(2*R)*0.5
	leftWheelVelocity = (2*user_lin_vel - user_ang_vel*L)/(2*R)*0.5
    
	print(rightWheelVelocity,leftWheelVelocity)
	p.setJointMotorControlArray(turtle1,[1,2], p.VELOCITY_CONTROL,targetVelocities = [leftWheelVelocity,rightWheelVelocity])
	p.stepSimulation()
	linV, angV = p.getBaseVelocity(turtle1)
	basePos, orn = p.getBasePositionAndOrientation(turtle1)
	# print("lin vel:", linV)
	# print("pos: ", basePos)
	yaw_change = p.getEulerFromQuaternion(orn)[2] - init_angle
	print(i, "yaw change:", yaw_change, lin_vel)
	#p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
	#p.setJointMotorControl2(turtle,2,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)
	

