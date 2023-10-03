import pybullet as p
import time
p.connect(p.GUI)
offset = [0,0,0]

turtle = p.loadURDF('/home/tushar-20-msi/turtlebot_stable_baseline/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro')
plane = p.loadURDF('/home/tushar-20-msi/turtlebot_stable_baseline/turtlebot3_description/urdf/simpleplane.urdf')
L = 0.16
R = 0.033
p.setRealTimeSimulation(1)

ang_vel = p.addUserDebugParameter('Angular_vel', -10, 10, 0)
lin_vel = p.addUserDebugParameter('linear_vel', -2, 2, 0)

position, orientation = p.getBasePositionAndOrientation(turtle)

for j in range (p.getNumJoints(turtle)):
	print(p.getJointInfo(turtle,j))
forward=0
turn=0
leftWheelVelocity=0
rightWheelVelocity=0
while (1):
	p.setGravity(0, 0, -10)
	time.sleep(1./1000)
	user_ang_vel = p.readUserDebugParameter(ang_vel)
	user_lin_vel = p.readUserDebugParameter(lin_vel)
	
	# p.setGravity(0,0,-10)
	# time.sleep(1./240.)
	# user_ang_vel = p.readUserDebugParameter(ang_vel)
    # user_lin_vel = p.readUserDebugParameter(lin_vel)
	# leftWheelVelocity=0
	#rightWheelVelocity=0
	# speed=10
	
	# for k,v in keys.items():
    #     if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
    #             turn = -0.5
    #     if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
    #             turn = 0
    #     if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
    #             turn = 0.5
    #     if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
    #             turn = 0

    #     if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
    #             forward=1
    #     if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
    #             forward=0
    #     if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
    #             forward=-1
    #     if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
    #             forward=0


	rightWheelVelocity = (2*user_lin_vel + user_ang_vel*L)/(2*R)
	leftWheelVelocity = (2*user_lin_vel - user_ang_vel*L)/(2*R)
        
	#print(rightWheelVelocity,leftWheelVelocity)
	p.setJointMotorControlArray(turtle,[1,2], p.VELOCITY_CONTROL,targetVelocities = [leftWheelVelocity,rightWheelVelocity])
	p.stepSimulation()
	
	#p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
	#p.setJointMotorControl2(turtle,2,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)
	

