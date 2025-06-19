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

class rotation_env(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               isDiscrete=False,
               init_angle = np.radians(140.99350876048027),
               init_pos = [0,0],
               renders=False,
               motion="CW",
               case = "1"):
    self._timeStep = 0.01
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._ballUniqueId = -1
    self._envStepCounter = 0
    self._renders = renders
    self._isDiscrete = isDiscrete
    self._init_angle = init_angle
    self._init_position = init_pos
    self._motion = motion
    self._case = case
    self._cam_dist = 4
    self._cam_yaw = 50
    self._cam_pitch = -35
    if self._renders:
      self._p = bc.BulletClient(connection_mode=pybullet.DIRECT)
    else:
      self._p = bc.BulletClient()

    # self.seed()
    # self.reset()
    observationDim = 2      # displacement, yaw_change, Lin_vel, ang_vel
    observation_high = np.ones(observationDim) * 10  #np.inf
    if (isDiscrete):
      self.action_space = spaces.Discrete(9)
    else:
      action_dim = 2                # Linear Velocity and Angular Velocity
      self._action_bound = 1
      action_high = np.array([self._action_bound]*action_dim)
      self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(-observation_high, observation_high, dtype=float)
    self.viewer = None

  def reset(self,seed = None):
    self._p.resetSimulation()
    #self._p.setPhysicsEngineParameter(numSolverIterations=300)
    self._p.setTimeStep(self._timeStep)
    self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self._p.loadURDF("plane.urdf")
    self._p.setGravity(0, 0, -10)
    self._robot = turtlebot3_burger.TurtleBot3(self._p, urdfRootPath=self._urdfRoot, timeStep=self._timeStep, init_angle = self._init_angle, case = self._case, init_position=self._init_position
                                               )
    self._envStepCounter = 0
    for i in range(100):
      self._p.stepSimulation()
    self._observation = self._robot.getObservation()
    self._initial_orientation = self._observation[1]
    self._robot_initial_pos = self._observation[0]
    self._observation[0] = 0    #initial displacement from initial position = 0
    self._observation[1] = 0    # initial yaw change = 0
    self._observation.pop()
    self._observation.pop()
    info = {}
    info['rew1'] = 0
    info['rew2'] = 0
    info['reward'] = 0
    info['ld'] = 0
    info['x'], info['y'] = self._robot_initial_pos[0], self._robot_initial_pos[1]
    info['yaw_change'] = 0
    info['orientation'] = self._initial_orientation
    return np.array(self._observation),info

  def __del__(self):
    self._p = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    if (self._renders):
      self.prev_ang_vel = self._p.getBaseVelocity(self._robot.robotUniqueId)[1][2]     # angular vel about z-axis before action is applied

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
      self.robot_observations()
      done = self._termination()
      if done:
        break
      self._envStepCounter += 1

    if self._motion=="CW":
      rew1, rew2 = self.CW_reward(action)
    elif self._motion=="CCW":
      rew1, rew2 = self.CCW_reward(action)
    reward = rew1 + rew2
    self._observation[0] = self.displacement
    self._observation[1] = self.yaw_change
    truncated = done
    info = {}
    info['rew1'] = rew1
    info['rew2'] = rew2
    info['reward'] = reward
    info['ld'] = self.displacement
    info['x'], info['y'] = self.robot_pos[0], self.robot_pos[1]
    info['yaw_change'] = self.yaw_change
    info['orientation'] = self.yaw
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

  def robot_observations(self,):
    robot_pos, robot_orn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
    self.robot_pos = robot_pos
    self.yaw = self._p.getEulerFromQuaternion(robot_orn)[2]
    if self._motion == "CW":
        if self.yaw < self._initial_orientation:
            self.yaw_change = - self.yaw + self._initial_orientation
        else:
            self.yaw_change = 2*math.pi - self.yaw + self._initial_orientation
        
    else:
        if self.yaw < self._initial_orientation:
            self.yaw_change = 2*math.pi + self.yaw - self._initial_orientation
        else:
            self.yaw_change = self.yaw - self._initial_orientation

    d = (abs(np.asarray(robot_pos) - np.asarray(self._robot_initial_pos)))
    self.displacement = math.sqrt(math.pow(d[0],2) + math.pow(d[1],2))

  def _termination(self):

    return self.displacement > 0.02 or self.yaw_change > 4

  def CW_reward(self, action):
    w_max = 2.84
    linV, angV = self._p.getBaseVelocity(self._robot.robotUniqueId)
    angVel = np.linalg.norm(angV)
    # print("angVel: ", angVel)
    if angVel <= 0.1*w_max:
      rew1 = -2000*angV[2]
    else:
      rew1 = 0
    rew2 = -4000*self.displacement
    # reward = rew1 + rew2

    return rew1, rew2
  
  def CCW_reward(self,action):
    w_max = 2.84
    linV, angV = self._p.getBaseVelocity(self._robot.robotUniqueId)
    angVel = np.linalg.norm(angV)
    # print("angVel: ", angVel)
    if angVel <= 0.1*w_max:
      rew1 = 100*angV[2]
    else:
      rew1 = 0
    rew2 = -100*self.displacement
    # reward = rew1 + rew2
    
    return rew1, rew2
  
  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
