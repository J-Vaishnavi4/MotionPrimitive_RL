#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from turtlebot3_burger_GymEnv import turtlebot3_burger_GymEnv
isDiscrete = False


def main():

  environment = turtlebot3_burger_GymEnv(renders=True, isDiscrete=isDiscrete)
  environment.reset()

  linear_Vel_slider = environment._p.addUserDebugParameter("linear_velocity", -1, 1, 0)
  angular_Vel_slider = environment._p.addUserDebugParameter("angular_velocity", -1, 1, 0)

  while (True):
    linear_Vel = environment._p.readUserDebugParameter(linear_Vel_slider)
    angular_Vel = environment._p.readUserDebugParameter(angular_Vel_slider)
    
    action = [linear_Vel, angular_Vel]
    state, reward, done, info = environment.step(action)
    obs = environment.getExtendedObservation()
    print("obs")
    print(obs)


if __name__ == "__main__":
  main()