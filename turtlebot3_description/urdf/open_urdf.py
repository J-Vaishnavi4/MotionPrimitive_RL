import pybullet as p
from time import sleep

p.connect(p.GUI)
p.setGravity(0, 0, -10)
ang_vel = p.addUserDebugParameter('Angular_vel', -2.84, 2.84, 0)
lin_vel = p.addUserDebugParameter('linear_vel', -0.22, 0.22, 0)
botId = p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/Gym-Medium-Post/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro', basePosition = [0, 0, 0])
plane = p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/Gym-Medium-Post/simple_driving/resources/simpleplane.urdf')
position, orientation = p.getBasePositionAndOrientation(botId)
print(p.getBasePositionAndOrientation(botId))
number_of_joints = p.getNumJoints(botId)

# for joint_number in range(number_of_joints):
#     info = p.getJointInfo(botId, joint_number)
#     print(info[0], ": ", info[1])
#sleep(3)

wheel_indices = [1, 2]
forward=0
turn=0
# hinge_indices = [0, 2]
speed=1


while True:
    leftWheelVelocity=0
    rightWheelVelocity=0	

    user_ang_vel = p.readUserDebugParameter(ang_vel)
    user_lin_vel = p.readUserDebugParameter(lin_vel)
    # p.setJointMotorControl2(botId,1,p.VELOCITY_CONTROL,targetVelocity=user_lin_vel)
    # p.setJointMotorControl2(botId,2,p.VELOCITY_CONTROL,targetVelocity=user_lin_vel)
    # if (user_ang_vel < 0):
    #     turn = -0.5
    # if (user_ang_vel == 0):
    #     turn = 0
    # if (user_ang_vel > 0):
    #     turn = 0.5

    if (user_lin_vel > 0 and user_ang_vel == 0):
        forward = 1
        turn = 0 
    if (user_lin_vel == 0 and user_ang_vel == 0):
        forward = 0
        turn = 0
    if (user_lin_vel < 0 and user_ang_vel == 0):
        forward = -1
        turn = 0

    rightWheelVelocity+= (forward+turn)*speed
    leftWheelVelocity += (forward-turn)*speed
    print(rightWheelVelocity , leftWheelVelocity)
	
    p.setJointMotorControl2(botId,1,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
    p.setJointMotorControl2(botId,2,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)

    # for joint_index in wheel_indices:
    #     p.setJointMotorControl2(botId, joint_index,
    #                             p.POSITION_CONTROL,
    #                             targetVelocity=user_lin_vel)
    # for joint_index in wheel_indices:
    #     p.setJointMotorControl2(botId, joint_index,
    #                             p.POSITION_CONTROL, 
    #                             targetPosition=user_ang_vel)
    p.stepSimulation()