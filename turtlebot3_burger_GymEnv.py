import os, inspect
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import math
import gym
import time
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import turtlebot3_burger
import random
from pybullet_utils import bullet_client as bc
import pybullet_data
from pkg_resources import parse_version

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class turtlebot3_burger_GymEnv(gym.Env):
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
      self._p = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._p = bc.BulletClient()

    self.seed()
    #self.reset()
    observationDim = 14 
    observation_high = np.ones(observationDim) * 100  #np.inf
    if (isDiscrete):
      self.action_space = spaces.Discrete(9)
    else:
      action_dim = 6 #consider everything as list
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(-observation_high, observation_high, dtype=float)
    self.viewer = None

  def reset(self):
    self._p.resetSimulation()
    #self._p.setPhysicsEngineParameter(numSolverIterations=300)
    self._p.setTimeStep(self._timeStep)
    # self._p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"))
    # stadiumobjects = self._p.loadSDF(os.path.join(self._urdfRoot, "stadium.sdf"))
    # # plane = self._p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"))
    # #move the stadium objects slightly above 0
    # for i in stadiumobjects:
    #   pos,orn = self._p.getBasePositionAndOrientation(i)
    #   newpos = [pos[0],pos[1],pos[2]-0.1]
    #   self._p.resetBasePositionAndOrientation(i,newpos,orn)

    # dist = 5 + 2. * random.random()
    # ang = 2. * 3.1415925438 * random.random()

    # ballx = dist * math.sin(ang)
    # bally = dist * math.cos(ang)
    # ballz = 1
    self._p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/MotionPrimitive_RL/turtlebot3_description/urdf/simpleplane.urdf')
    self._goalUniqueId = self._p.loadURDF('/home/vaishnavi/Documents/IISc/Car-Plane robot_RL/MotionPrimitive_RL/turtlebot3_description/urdf/simplegoal.urdf', [2, 0, 0])
    self._p.setGravity(0, 0, -10)
    self._robot = turtlebot3_burger.TurtleBot3(self._p, urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    for i in range(100):
      self._p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

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
    if (self._renders):
      basePos, orn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
      #self._p.resetDebugVisualizerCamera(1, 30, -40, basePos)
      d = (abs(np.asarray(basePos)[0] - np.asarray(self._p.getBasePositionAndOrientation(self._goalUniqueId)[0])))
      self.prev_dist_to_goal = math.sqrt(math.pow(d[0],2) + math.pow(d[1],2))
      self.prev_dist_orn = self._p.getEulerFromQuaternion(orn)[2]
      self.prev_vel = self._p.getBaseVelocity(self._robot.robotUniqueId)[0]
      # print("before action")
      # print(self.prev_dist_to_goal)




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
      #if self._renders:
      time.sleep(self._timeStep)
      self._observation = self.getExtendedObservation()
      #print(self._observation)

      if self._termination():
        break
      self._envStepCounter += 1
    reward = self._reward()
    done = self._termination()
    #print("len=%r" % len(self._observation))

    return np.array(self._observation), reward, done, {}

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
    #closestPoints = self._p.getClosestPoints(self._robot.robotUniqueId, self._goalUniqueId,10000)
    d = (abs(np.asarray(self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)[0]) - np.asarray(self._p.getBasePositionAndOrientation(self._goalUniqueId)[0])))
    dist = math.pow(d[0],2) + math.pow(d[1],2) + math.pow(d[2],2)
    yaw = self._p.getEulerFromQuaternion(self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)[1])[2]
    return self._envStepCounter > 10000 or dist < 0.01 or dist > 1.2 or  yaw > 0.5 or yaw < -0.5

  def _reward(self):
    # closestPoints = self._p.getClosestPoints(self._robot.robotUniqueId, self._ballUniqueId,
    #                                          10000)
    d = (abs(np.asarray(self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)[0]) - np.asarray(self._p.getBasePositionAndOrientation(self._goalUniqueId)[0])))
    dist_to_goal = math.sqrt(math.pow(d[0],2) + math.pow(d[1],2) + math.pow(d[2],2))
    # print("after action")
    # print(dist_to_goal)

    yaw = self._p.getEulerFromQuaternion(self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)[1])[2]
    lV, aV = self._p.getBaseVelocity(self._robot.robotUniqueId)
    

    if (dist_to_goal<=0.01):
      print("reached")
      reward = 500

    else:
      rew1 = 100*(self.prev_dist_to_goal - dist_to_goal)
      #print(yaw)
      rew2 = -100*abs(yaw)
      rew3 = 0*(dist_to_goal < 0.2 and (0,0,0)<lV < self.prev_vel) 
      rew4 = -50 *(lV<(0,0,0))
      reward = rew1 + rew2 + rew3 

    #print(reward)
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step