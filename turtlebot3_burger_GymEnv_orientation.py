import os, inspect
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import math
import gymnasium as gym
import time
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pybullet
import turtlebot3_burger
import random
from pybullet_utils import bullet_client as bc
import pybullet_data
from pkg_resources import parse_version

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class turtlebot3_burger_GymEnv_orientation(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=50,
               isEnableSelfCollision=True,
               isDiscrete=False,
               renders=False):
    #print("init")
    self._timeStep = 0.01
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._ballUniqueId = -1
    self._envStepCounter = 0
    self._renders = renders
    self._isDiscrete = isDiscrete
    self._cam_dist = 4
    self._cam_yaw = 50
    self._cam_pitch = -35
    if self._renders:
      self._p = bc.BulletClient(connection_mode=pybullet.DIRECT)
    else:
      self._p = bc.BulletClient()

    self.seed()
    #self.reset()
    observationDim = 14
    observation_high = np.ones(observationDim) * 100  #np.inf
    if (isDiscrete):
      self.action_space = spaces.Discrete(9)
    else:
      action_dim = 2 #consider everything as list
      self._action_bound = 1
      # action_high = np.array([self._action_bound]*action_dim)
      self.action_space = spaces.Box(np.array([0,-self._action_bound]), np.array([0,self._action_bound]), dtype=np.float32)
    self.observation_space = spaces.Box(-observation_high, observation_high, dtype=float)
    self.viewer = None

  def reset(self,seed = None):
    self._p.resetSimulation()
    #self._p.setPhysicsEngineParameter(numSolverIterations=300)
    self._p.setTimeStep(self._timeStep)
    self._offset_to_goal = [0, 0, 0]
    self._euler_orientation = [0, 0, -1*math.pi/3]
    self._p.loadURDF(currentdir+'/turtlebot3_description/urdf/simpleplane.urdf')
    self._goal_orientation = self._p.getQuaternionFromEuler(self._euler_orientation)
    self._goalUniqueId = self._p.loadURDF(currentdir+'/turtlebot3_description/urdf/simplegoal.urdf', basePosition = self._offset_to_goal, baseOrientation = self._goal_orientation)
    self._p.setGravity(0, 0, -10)
    self._robot = turtlebot3_burger.TurtleBot3(self._p, urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    for i in range(100):
      self._p.stepSimulation()
    self._observation = self.getExtendedObservation()
    info = {}
    return np.array(self._observation),info

  def __del__(self):
    self._p = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    self._observation = self._robot.getObservation()
    robotPos, robotOrn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
    goalPos, goalOrn = self._p.getBasePositionAndOrientation(self._goalUniqueId)
    invRobotPos, invRobotOrn = self._p.invertTransform(robotPos, robotOrn)
    goalPosInRobot, goalOrnInRobot = self._p.multiplyTransforms(invRobotPos, invRobotOrn, goalPos, goalOrn)

    self._observation.extend([goalPosInRobot[0], goalPosInRobot[1]])
    # print(self._observation, type(ballPosInRobot[0]))
    return self._observation

  def step(self, action):
    # print(action)
    if (self._renders):
      basePos, orn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
      d = (abs(np.asarray(basePos)[0] - np.asarray(self._offset_to_goal)))
      self.prev_orn = self._p.getEulerFromQuaternion(orn)[2]
      self.prev_orn_diff = abs(self.prev_orn - self._euler_orientation[2])
      self.prev_ang_vel = self._p.getBaseVelocity(self._robot.robotUniqueId)[1][2]    #ang vel about z-axis

    if (self._isDiscrete):
      rightVel = [-1, -0.5, -0.1, 0, 0.5, 0.1, 1, 0.2, 0.8]
      leftVel = [-1, -0.5, -0.1, 0, 0.5, 0.1, 1, 0.2, 0.8]
      rightCmd = rightVel[action]
      leftCmd = leftVel[action]
      realaction = [rightCmd, leftCmd]
    else:
      realaction = action

    self._robot.applyAction(realaction[0],realaction[1])
    for i in range(self._actionRepeat):
      self._p.stepSimulation()
      time.sleep(self._timeStep)
      self._observation = self.getExtendedObservation()

      if self._termination():
        break
      self._envStepCounter += 1
    reward = self._reward()
    done = self._termination()
    truncated = done
    return np.array(self._observation), reward, done, truncated, {}

  def render(self, mode='human', close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos, orn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    robot_pos, robot_orn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
    # d = (abs(np.asarray(robot_pos) - np.asarray(self._offset_to_goal)))
    # dist = math.sqrt(math.pow(d[0],2) + math.pow(d[1],2))
    goal_orientation = self._euler_orientation[2]
    yaw = self._p.getEulerFromQuaternion(robot_orn)[2]
    # print("yaw: ",yaw)
    lV, aV = self._p.getBaseVelocity(self._robot.robotUniqueId)
    return self._envStepCounter > 5000 or (aV[2]<=0.01 and abs(yaw - goal_orientation)<0.01) #or dist>=0.01 or abs(yaw - goal_orientation) <= 0.01

  def _reward(self):

    robot_pos,robot_orn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
    d = (abs(np.asarray(robot_pos) - np.asarray(self._offset_to_goal)))
    dist_to_goal = math.sqrt(math.pow(d[0],2) + math.pow(d[1],2))
    goal_orientation = self._euler_orientation[2]
    yaw = self._p.getEulerFromQuaternion(robot_orn)[2]
    current_orn_diff = abs(yaw - goal_orientation)
    # print(yaw - goal_orientation)
    lV, aV = self._p.getBaseVelocity(self._robot.robotUniqueId)
    # if dist_to_goal!=0:
    #   print("dist to goal:", dist_to_goal)

    if (current_orn_diff<=0.01) and abs(aV[2]) < 0.1:
      print("reached")

    rew1 = -100*(dist_to_goal)                                    # penalizing linear displacement from initial position
    rew2 = 100*(current_orn_diff<0.1)*(self.prev_ang_vel - aV[2])        # for reduction in angular velocity as it approaches goal orientation
    rew3 = 100*(self.prev_orn_diff - current_orn_diff)            # positive reward for reduced orn difference
    rew4 = 5000*((current_orn_diff<=0.01) and abs(aV[2]) < 0.1)   # reward when robot's orientation is close to goal and it's ang vel is small
    # rew5 to account for clockwise and anti clockwise rotation of robot based on yaw and goal orientation 
    if 0 < yaw - goal_orientation < math.pi or yaw-goal_orientation<-math.pi:                
      rew5 =  -10*aV[2]
    elif -math.pi < yaw - goal_orientation < 0 or yaw - goal_orientation >math.pi:
      rew5 = 10*aV[2]
    else:
      rew5=0
    reward = rew1 + rew2 + rew3 + rew4 + rew5
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
