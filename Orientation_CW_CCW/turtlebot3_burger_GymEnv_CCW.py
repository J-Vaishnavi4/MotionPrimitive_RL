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

class turtlebot3_burger_GymEnv_CCW(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
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
    # self.reset()
    observationDim = 8      # displacement, yaw_change, Vx,Vy, Vz, Wx, Wy, Wz
    observation_high = np.ones(observationDim) * 10  #np.inf
    if (isDiscrete):
      self.action_space = spaces.Discrete(9)
    else:
      action_dim = 2             # Linear Velocity and Angulr Velocity
      self._action_bound = 1
      action_high = np.array([self._action_bound]*action_dim)
      self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(-observation_high, observation_high, dtype=float)
    self.viewer = None

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
    self._observation[0] = 0    # initial displacement from initial position = 0
    self._observation[1] = 0    # initial yaw change=0
    info = {}
    return np.array(self._observation),info

  def __del__(self):
    self._p = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    self._observation = self._robot.getObservation()
    # print(self._observation)
    return self._observation

  def step(self, action):
    if (self._renders):
      self.prev_ang_vel = self._p.getBaseVelocity(self._robot.robotUniqueId)[1][2]    # ang vel about z-axis before action is applied

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
    reward, yaw_change, displacement = self._reward()
    self._observation[0] = displacement
    self._observation[1] = yaw_change
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
    d = (abs(np.asarray(robot_pos) - np.asarray(self._robot_initial_pos)))
    displacement = math.sqrt(math.pow(d[0],2) + math.pow(d[1],2))
    if displacement>=0.15:
      print("terminated because of displacement")
    return displacement>=0.15 or self._envStepCounter>5000

  def _reward(self):
    robot_pos,robot_orn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
    d = (abs(np.asarray(robot_pos) - np.asarray(self._robot_initial_pos)))
    displacement = math.sqrt(math.pow(d[0],2) + math.pow(d[1],2))
    yaw = self._p.getEulerFromQuaternion(robot_orn)[2]
    if yaw < self._initial_orientation:
      yaw_change = 2*math.pi + yaw - self._initial_orientation
    else:
      yaw_change = yaw - self._initial_orientation
    lV, aV = self._p.getBaseVelocity(self._robot.robotUniqueId)

    " rew1: Penalize linear displacement from robot's initial position"
    " rew2: Angular velocity should be positive for counter clockwise rotation"
    " rew3: Robot should slow down as it is close to completing the 360 degree rotation"

    rew1 = -1000*(displacement)                               # penalizing linear displacement from initial position
    rew2 = 1000*(yaw_change)*(aV[2])*(yaw_change <= 0.7*2*math.pi) + 500*(yaw_change)*(aV[2])*(0.7*2*math.pi < yaw_change<=0.85*2*math.pi) + (50*yaw_change)*(aV[2])*(0.85*2*math.pi < yaw_change <= 2*math.pi)
    rew3 = (0.85*2*math.pi < yaw_change < 2*math.pi)*(aV[2])*(abs(self.prev_ang_vel)-abs(aV[2]))*1000

    reward = rew1 + rew2 + rew3
    return reward, yaw_change, displacement

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
