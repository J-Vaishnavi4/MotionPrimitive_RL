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
import csv


RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class turtlebot3_burger_GymEnv_backward(gym.Env):
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
    self.reset()
    observationDim = 4      # displacement, yaw_change, Lin_vel, Ang_vel
    observation_high = np.ones(observationDim) * 10  #np.inf
    if (isDiscrete):
      self.action_space = spaces.Discrete(9)
    else:
      action_dim = 2 #consider everything as list
      self._action_bound = 1
      action_high = np.array([self._action_bound]*action_dim)
      self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(-observation_high, observation_high, dtype=float)
    self.viewer = None
    # if not os.path.exists("./reward/"):
    #    os.makedirs("./reward/")
    # with open("./reward/reward_BMP_10.csv",'w',newline='') as file:
    #   writer = csv.writer(file)

  def reset(self,seed = None):
    self._p.resetSimulation()
    #self._p.setPhysicsEngineParameter(numSolverIterations=300)
    self._p.setTimeStep(self._timeStep)

    self._p.loadURDF(currentdir+'/turtlebot3_description/urdf/simpleplane.urdf')
    self._p.setGravity(0, 0, -10)
    self._robot = turtlebot3_burger.TurtleBot3(self._p, urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    for i in range(100):
      self._p.stepSimulation()
    self._observation = self.getExtendedObservation()
    self._robot_initial_pos = self._observation[0]
    self._initial_orientation = self._observation[1]

    # print("init theta: ",self._initial_orientation)
    self._observation[0] = 0  # initial displacement = 0
    self._observation[1] = 0  # initial yaw_change = 0
    
    self.reward_value=0
    info = {}
    info['rew1']=0
    info['rew2']=0

    # info['rew3']=0
    return np.array(self._observation),info

  def __del__(self):
    self._p = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    self._observation = self._robot.getObservation()
    return self._observation

  def step(self, action):
    if (self._renders):
      basePos, orn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
      d = (abs(np.asarray(basePos)[0] - np.asarray(self._robot_initial_pos)))
      self._prev_speed = self._observation[2]

    if (self._isDiscrete):
      rightVel = [-1, -0.5, -0.1, 0, 0.5, 0.1, 1, 0.2, 0.8]
      leftVel = [-1, -0.5, -0.1, 0, 0.5, 0.1, 1, 0.2, 0.8]
      rightCmd = rightVel[action]
      leftCmd = leftVel[action]
      realaction = [rightCmd, leftCmd]
    else:
      realaction = action

    self._robot.applyAction(-realaction[0],-realaction[1])
    for i in range(self._actionRepeat):
      self._p.stepSimulation()
      time.sleep(self._timeStep)
      self._observation = self.getExtendedObservation()

      if self._termination():
        break
      self._envStepCounter += 1
    # print(self._envStepCounter)
    rew1, rew2, yaw_change, displacement= self._reward(action)
    reward = min(rew1, rew2)
    # reward = rew1*rew2
    # self.reward_value += reward
    self._observation[0] = displacement
    self._observation[1] = yaw_change
    done = self._termination()
    # if done:
    #   with open("./reward/reward_BMP_10.csv",'a',newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(np.array([self.reward_value]))
    #     file.close()
    #   self.reward_value = 0
    # if yaw_change>0.1:
    #    print("displacement: ", displacement, yaw_change)
    truncated = done
    info = {}
    info['rew1']=rew1
    info['rew2']=rew2
    # info['rew3']=rew3
    return np.array(self._observation), reward, done, truncated, info

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
    yaw = self._p.getEulerFromQuaternion(robot_orn)[2]
    yaw_change = abs(abs(yaw) - abs(self._initial_orientation))
    return self._envStepCounter>10000 or yaw_change>0.1

  def _sign_value(self,yaw):
    if abs(yaw) < math.pi/2:
      c1 = -1
      if yaw > 0:
        c2 = -1
      elif yaw < 0:
        c2 = 1
      else:
        c2 = 0
    elif abs(yaw) > math.pi/2:
      c1 = 1
      if yaw > 0:
        c2 = -1
      elif yaw < 0:
        c2 = 1
      else:
        c2 = 0
    else:
      c1 = 0
      if yaw > 0:
        c2 = -1
      elif yaw < 0:
        c2 = 1

    return c1, c2

  def _reward(self, action):
    robot_pos,robot_orn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
    d = (abs(np.asarray(robot_pos) - np.asarray(self._robot_initial_pos)))
    displacement = math.sqrt(math.pow(d[0],2) + math.pow(d[1],2))
    yaw = self._p.getEulerFromQuaternion(robot_orn)[2]
    yaw_change = abs(abs(yaw) - abs(self._initial_orientation))

    rew1 = -action[0]
    rew2 = 1/(1+math.exp(100*yaw_change)) 

    return rew1, rew2, yaw_change, displacement

  def _reward2(self,action):
    robot_pos,robot_orn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
    d = (abs(np.asarray(robot_pos) - np.asarray(self._robot_initial_pos)))
    displacement = math.sqrt(math.pow(d[0],2) + math.pow(d[1],2))
    yaw = self._p.getEulerFromQuaternion(robot_orn)[2]
    yaw_change = abs(abs(yaw) - abs(self._initial_orientation))
    lV, aV = self._p.getBaseVelocity(self._robot.robotUniqueId)

    rew1 = -action[0]
    rew2 = 4#1*int(yaw_change <= 0.05) - 1*int(yaw_change > 0.05)
    return rew1, rew2, yaw_change, displacement
  

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
