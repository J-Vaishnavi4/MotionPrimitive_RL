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

class translation_env(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               isDiscrete=False,
               renders=False,
               init_angle = np.radians(70),
               init_pos = [0,0],
               motion="F",
               case = "1"):
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
    self._motion = motion
    self._case = case
    self._cam_dist = 4
    self._cam_yaw = 50
    self._cam_pitch = -35
    self._init_angle = init_angle
    self._init_pos = init_pos
    self._lindisp = 0# np.array([0.0,0.0,0.0])
    if self._renders:
      self._p = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._p = bc.BulletClient()
    # self.seed()
    #self.reset()
    observationDim = 4      # displacement, lateral disp, Lin_vel in x, Lin_vel in y
    observation_high = np.ones(observationDim) * 10  #np.inf
    if (isDiscrete):
      self.action_space = spaces.Discrete(9)
    else:
      action_dim = 2
      self._action_bound = 1
      action_high = np.array([self._action_bound]*action_dim)
      self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(-observation_high, observation_high, dtype=float)
    self.viewer = None

  def reset(self,seed = None):
    self._p.resetSimulation()
    self._p.setTimeStep(self._timeStep)

    self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self._p.loadURDF("plane.urdf")
    self._p.setGravity(0, 0, -10)
    # print("init_angle", np.rad2deg(self._init_angle))
    self._robot = turtlebot3_burger.TurtleBot3(self._p, urdfRootPath=self._urdfRoot, timeStep=self._timeStep, init_angle = self._init_angle, case = self._case, init_position=self._init_pos)
    self._envStepCounter = 0
    for i in range(100):
      self._p.stepSimulation()
    self._observation = self._robot.getObservation()
    self._robot_initial_pos = self._observation[0]
    self._initial_orientation = self._observation[1]
    self._observation[0]=0  # initial displacement = 0
    self._observation[1]=0  # initial ld = 0
    self._lindisp = 0
    info = {}
    info['rew1']=0
    info['rew2']=0
    info['ld']=0
    info['x'] = self._robot_initial_pos[0]
    info['y'] = self._robot_initial_pos[1]
    info['yaw_change'] = 0
    info['orientation'] = self._initial_orientation
    return np.array(self._observation), info

  def __del__(self):
    self._p = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    if (self._renders):
      basePos, orn = self._p.getBasePositionAndOrientation(self._robot.robotUniqueId)
      d = (abs(np.asarray(basePos)[0] - np.asarray(self._robot_initial_pos)))
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
      self.prev_disp = self._lindisp
      self.robot_observations()
      
      done = self._termination()
      if done:
        break
      self._envStepCounter += 1      
    
    if self._motion=="Forward":
      rew1, rew2, yaw_change, lateral_deviation, displacement, robot_position = self.forward_reward(action)
    elif self._motion=="Backward":
      rew1, rew2, yaw_change, lateral_deviation, displacement, robot_position = self.backward_reward(action)
    
    reward = rew1 + rew2 #min(rew1, rew2)
    self._observation[0] = displacement 
    self._observation[1] = lateral_deviation

    truncated = done
    info = {}
    info['rew1']=rew1
    info['rew2']=rew2
    info['ld']=lateral_deviation
    info['x'] = robot_position[0]
    info['y'] = robot_position[1]
    info['yaw_change'] = yaw_change
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
    self.yaw_change = abs(abs(self.yaw) - abs(self._initial_orientation))

    d = (abs(np.asarray(robot_pos) - np.asarray(self._robot_initial_pos)))
    self.displacement = math.sqrt(math.pow(d[0],2) + math.pow(d[1],2))

    linVw, angVw = self._p.getBaseVelocity(self._robot.robotUniqueId)
    rotn_mat = np.array(self._p.getMatrixFromQuaternion(robot_orn)).reshape(3,3)

    self.linVb = np.dot(rotn_mat.T,np.array(linVw))
    angVb = np.dot(rotn_mat.T,np.array(angVw))
    print( "linV_world: ", np.linalg.norm(linVw))
    self._lindisp += np.linalg.norm(self.linVb)*self._timeStep
    
    self.theta = math.atan2((robot_pos[1]-self._robot_initial_pos[1]),(robot_pos[0]-self._robot_initial_pos[0]))
    self.alpha = self._initial_orientation - self.theta
    self.lateral_deviation = abs(self._lindisp*math.sin(self.alpha))
    # print("disp: ", self.displacement, self._lindisp)

  def _termination(self):

    return self._lindisp > 1 or self.lateral_deviation > 0.02 or self.yaw_change > 0.2 
  
  def forward_reward(self, action):
    v_max = 0.22*0.2
    # linVb = np.linalg.norm(self.linVb)
    linVbx = self.linVb[0]
    delta_disp = self._lindisp - self.prev_disp
    # print("linVbx : ", linVbx)
    if linVbx <= v_max:
      rew1 =30*self.linVb[0] + 10*np.exp(self._lindisp)
    else:
      rew1 = 0

    # rew1 = 50*self.linVb[0]/(abs(self.linVb[0]))
    rew2 = -100*self.lateral_deviation*(self.lateral_deviation>=0.01) + 0.01*(self.lateral_deviation<0.01)/(self.lateral_deviation+0.01) #- 10*self.yaw_change
    # print("rew ", rew1, rew2)
    return rew1, rew2, self.yaw_change, self.lateral_deviation, self._lindisp, self.robot_pos
  
  def backward_reward(self, action):
    v_max = -0.22*1
    # linVb = np.linalg.norm(self.linVb)
    linVbx = self.linVb[0]
    delta_disp = self._lindisp - self.prev_disp
    # print("linVbx : ", linVbx)
    if linVbx >= v_max:
      rew1 = -30*self.linVb[0] + 10*np.exp(self._lindisp)
    else:
      rew1 = 0

    # print("cal vel: ", V,w,"norms: ", np.linalg.norm(V), np.linalg.norm(w))
    # rew1 = -50*self.linVb[0]/(abs(self.linVb[0]))
    rew2 = -100*self.lateral_deviation*(self.lateral_deviation>=0.01) + 0.1*(self.lateral_deviation<0.01)/(self.lateral_deviation+0.01)
    # print("rew ", rew1, rew2)
    # print(linVbx, self.linVb[0], v_max)
    return rew1, rew2, self.yaw_change, self.lateral_deviation, self.displacement, self.robot_pos

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
