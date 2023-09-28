import pybullet as p
import time
import math

p.connect(p.GUI)
p.setGravity(0,0,-10)
offset = [0,0,0]
turtle = p.loadURDF('//home/vaishnavi/Documents/IISc/RL/Coding/MotionPrimitive_RL_1/turtlebot3_description/urdf/Edit_turtlebot3_burger.urdf.xacro',offset)
# turtle = p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/MotionPrimitive_RL/turtlebot3_description/urdf/Edit_turtlebot3_burger.urdf.xacro',offset)
plane = p.loadURDF('/home/vaishnavi/Documents/IISc/RL/Coding/MotionPrimitive_RL_1/simple_driving/resources/simpleplane.urdf')
p.setRealTimeSimulation(1)

for j in range (p.getNumJoints(turtle)):
	print(p.getJointInfo(turtle,j))

def apply_action(action):
        # Expects action to be two dimensional
        leftWheelVelocity,rightWheelVelocity = action[0],action[1]
        p.setJointMotorControlArray(turtle,[1,2],p.VELOCITY_CONTROL, targetVelocities=[leftWheelVelocity,rightWheelVelocity],forces=[10,10])
        vel = p.getBaseVelocity(turtle)[0]
        print("Vel: ",leftWheelVelocity, rightWheelVelocity,vel)
        # p.setJointMotorControl2(turtle,2,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)
        # p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
leftWheelVelocity = 0
rightWheelVelocity = 0
while(1):
        time.sleep(1./240.)
        p.stepSimulation()
        for i in range(4):
                print("i: ",i)
                action = [100*math.cos(i/2),100*math.cos(i/2)]
                apply_action(action)
