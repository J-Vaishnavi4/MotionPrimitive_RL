import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared

from Gym_env import Gym_environment
from multiprocessing import Pool
import datetime
from stable_baselines3 import ppo, SAC
from stable_baselines3.common.env_checker import check_env
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math, time, csv


class prediction_model:
    def __init__(self, 
               case="1",
               node_x_target=0, 
               node_y_target=0, 
               robot_x=0, 
               robot_y=0, 
               robot_angle=0, 
               time_target=0):
        
        self.x_target_node = node_x_target
        self.y_target_node = node_y_target
        self.x_robot = robot_x
        self.y_robot = robot_y
        self.robot_angle = robot_angle

        path_slope = np.arctan2((self.y_target_node-self.y_robot),(self.x_target_node - self.x_robot))
        self.required_displacement = np.linalg.norm([self.x_target_node-self.x_robot, self.y_target_node-self.y_robot])
        self.required_yaw_change = - path_slope + self.robot_angle
        self.case = case
        total_predicted_time = 0
        self.time_req = time_target
        self.list_of_models = []
        lin_model_nums = [10,9,8,7,6,5,4,3,2]
        rtn_model_nums = [10,6,2]
        self.list = [10,6,2]
        if self.case == "1":
            self.case = "Ideal"
        elif self.case == "2":
            self.case = "Diff_wheel_radius"
        elif self.case == "3":
            self.case = "Diff_max_wheel_vel"
        else:
            raise SystemExit("Incorrect self.case number")

        self.lin_GP_models = []
        self.lin_policy_models = []
        self.CW_GP_models = []
        self.CW_policy_models=[]
        self.CCW_GP_models = []
        self.CCW_policy_models=[]
        begin = time.time()
        for model_num_lin in lin_model_nums:
            self.lin_GP_models.append(pickle.load(open("./"+self.case+"/GP_models/F"+str(model_num_lin)+"_noise.dump", "rb")))
            self.lin_policy_models.append(ppo.PPO.load(os.path.join(currentdir,"./"+self.case+"/Policies/F_"+str(model_num_lin))))
        for model_num_rtn in rtn_model_nums:
            self.CW_GP_models.append(pickle.load(open("./"+self.case+"/GP_models/CW"+str(model_num_rtn)+"_noise.dump", "rb")))
            self.CW_policy_models.append(ppo.PPO.load(os.path.join(currentdir,"./"+self.case+"/Policies/CW_"+str(model_num_rtn))))
        # for model_num_rtn in rtn_model_nums:
            self.CCW_GP_models.append(pickle.load(open("./"+self.case+"/GP_models/CCW"+str(model_num_rtn)+"_noise.dump", "rb")))
            self.CCW_policy_models.append(ppo.PPO.load(os.path.join(currentdir,"./"+self.case+"/Policies/CCW_"+str(model_num_rtn))))
        print("Time to load GP models: ", time.time()-begin)


    def reset(self, 
               case="1",
               node_x_target=0, 
               node_y_target=0, 
               robot_x=0, 
               robot_y=0, 
               robot_angle=0, 
               time_target=0):
        
        self.x_target_node = node_x_target
        self.y_target_node = node_y_target
        self.x_robot = robot_x
        self.y_robot = robot_y
        self.robot_angle = robot_angle

        path_slope = np.arctan2((self.y_target_node-self.y_robot),(self.x_target_node - self.x_robot))
        self.required_displacement = np.linalg.norm([self.x_target_node-self.x_robot, self.y_target_node-self.y_robot])
        self.required_yaw_change = - path_slope + self.robot_angle
        self.case = case
        total_predicted_time = 0
        self.time_req = time_target
        self.list_of_models = []

    # def linear_time_predict(self, lin_GP_models):
    #     lin_predict_time, lin_std_prediction = lin_GP_models.predict(np.array([self.required_displacement]).reshape(1,-1), return_std=True)
    #     lin_predict_time = round(lin_predict_time[0][0])
    #     return lin_predict_time

    def predict_model(self):

        lin_time_list = []
        if self.required_yaw_change==0:
            model_num_rotn = 0
            MP_rotn = 0
            rotn_predict_time = 0
            if self.required_displacement>0 and len(lin_time_list)==0:
                MP_name = "F"
                MP_lin = 3
                for model_lin in self.lin_GP_models:                
                    # start_time_lin = time.time() 
                    lin_predict_time, lin_std_prediction = model_lin.predict(np.array([self.required_displacement]).reshape(1,-1), return_std=True)
                    # print("predict: ", time.time()-start_time_lin)
                    lin_predict_time = round(lin_predict_time[0][0])
                    # print("Linear time: ", self.time_req, lin_predict_time, model_lin)
                    total_predicted_time = lin_predict_time
                    if abs(total_predicted_time-self.time_req)<150:
                        # model_num_lin = 10-self.lin_GP_models.index(model_lin)
                        break
                model_num_lin = 10-self.lin_GP_models.index(model_lin)
                # print("start time lin: ", time.time()-start_time_lin)
        else:
            if 0< self.required_yaw_change <= math.pi:
                MP_name = "CW"
                MP_rotn = 1
                rotn_models = self.CW_GP_models
            else:
                MP_name = "CCW"
                MP_rotn = 2
                self.required_yaw_change = - self.required_yaw_change
                rotn_models = self.CCW_GP_models
            
            for model_rotn in rotn_models:
                rotn_predict_time, rotn_std_prediction = model_rotn.predict(np.array([self.required_yaw_change]).reshape(1,-1), return_std=True)
                rotn_predict_time = round(rotn_predict_time[0][0])
                
                if self.required_displacement>0 and len(lin_time_list)==0:
                    MP_name = "F"
                    MP_lin = 3
                    for model_lin in self.lin_GP_models:
                        lin_predict_time, lin_std_prediction = model_lin.predict(np.array([self.required_displacement]).reshape(1,-1), return_std=True)
                        lin_predict_time = round(lin_predict_time[0][0])
                        total_predicted_time = rotn_predict_time + lin_predict_time
                        # print("Linear time: ",  self.time_req, lin_predict_time, model_lin)
                        if abs(total_predicted_time-self.time_req)<150:
                            # model_num_lin = 10-self.lin_GP_models.index(model_lin)
                            break
                        # else:
                            # model_num_lin = self.lin_GP_models.index(model_lin)
                        lin_time_list.append(lin_predict_time)
                    model_num_lin = 10-self.lin_GP_models.index(model_lin)
                elif len(lin_time_list)!=0:
                    # print("Linear model List created")
                    total_time_list = [x + rotn_predict_time for x in lin_time_list]
                
                    closest_time_cal =  min(total_time_list, key=lambda x:abs(x-self.time_req))
                    if closest_time_cal<150:
                        index = total_time_list.index(closest_time_cal)
                        lin_predict_time = lin_time_list[index]
                        model_num_lin = 10-index
                        break
                
                if abs(total_predicted_time-self.time_req)<150:
                    
                    model_num_rotn = self.list[rotn_models.index(model_rotn)]
                    break
                else:
                    model_num_rotn = self.list[rotn_models.index(model_rotn)]
                    continue
        

        self.list_of_models.append([MP_rotn, model_num_rotn, rotn_predict_time, MP_lin, model_num_lin, lin_predict_time])
        compensation_time = total_predicted_time - self.time_req
        return self.list_of_models[0], 0

    def apply_model(self,env, MP_num, model_num, time_steps, pos_0, angle_0,start_time):
        # print("inputs: ", env, MP_num, model_num, time_steps)
        # begin=time.time()
        case = "1"
        if MP_num == 1:
            MP_name = "CW"
            obs,info = env.reset_MP(motion = MP_name, init_angle = angle_0, init_position=[pos_0[0], pos_0[1]])
            # print((self.list).index(model_num))
            model = self.CW_policy_models[(self.list).index(model_num)]
        elif MP_num == 2:
            MP_name = "CCW"
            obs,info = env.reset_MP(motion = MP_name, init_angle = angle_0, init_position=[pos_0[0], pos_0[1]])
            # print("index: ",(self.list).index(model_num), self.CCW_policy_models)
            model = self.CCW_policy_models[(self.list).index(model_num)]
        elif MP_num == 3:
            MP_name = "F"
            obs,info = env.reset_MP(motion = MP_name, init_angle = angle_0, init_position=[pos_0[0], pos_0[1]])
            model = self.lin_policy_models[10-model_num]
        # elif MP_num == 4:
        #     MP_name = "B"
        #     env = translation_env(renders=True, isDiscrete=False, motion="Backward", self.case = self.case,init_angle = angle_0, init_pos=[pos_0[0], pos_0[1]])
        # print("time to reset model: ", time.time()-begin)
        if case =="1":
            case = "Ideal"
        ld, x, y = [info['ld']], [info['x']], [info['y']]
        
        # begin=time.time()
        for j in range(time_steps+1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done,truncated, info = env.step(action)
            ld.append(info['ld'])
            x.append(info['x'])
            y.append(info['y'])
            env.render(mode='human')
            # if MP_name =="F":
            # with open(os.path.join(currentdir,"./"+case)+'/new_traj_two_target1.csv', 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow([info['x'], info['y'], time.time()-start_time])
        # print("time to apply actions: ", time.time()-begin)
        return info['x'], info['y'], info['orientation']

def main():
    # nodes = [(0.5, 0.5, 0),
    # (0.6414213562373096, 0.6414213562373096, 5.471111777980823),
    # (0.8313979816990338, 0.7039424069176664, 9.442909413535396),
    # (0.9675247443543284, 0.8504671641738689, 13.147102090077595),
    # (1.1036515070096231, 0.9969919214300714, 14.694445416654737),
    # (1.2397782696649178, 1.143516678686274, 15.56531318507691),
    # (1.3759050323202124, 1.2900414359424766, 16.467115088986162),
    # (1.512031794975507, 1.436566193198679, 17.402109046549526),
    # (1.6481585576308018, 1.5830909504548816, 18.292572749609132),
    # (1.7842853202860964, 1.729615707711084, 19.409158662840863),
    # (1.920412082941391, 1.8761404649672866, 20.79635135387038),
    # (2.0565388455966858, 2.022665222223489, 21.60778670895039),
    # (2.1926656082519806, 2.1691899794796914, 22.708422121939968),
    # (2.3287923709072755, 2.315714736735894, 23.627634948227005),
    # (2.46491913356257, 2.4622394939920964, 24.93196337727993)
    # ]  

    nodes = [(0.5 ,0.5,0),
             (1.15, 0.58,9.3),
             (1.51 ,0.8,11.4),
             (1.82 ,0.6, 13.2),
             (2.08, 0.59, 17.7),
             (2.33, 0.51, 27.9),
             (2.1 ,1.02, 30.9),
             (1.65, 1.24, 35.2),
             (1.49 ,1.48,36.7),
             (1.11, 1.58, 40.3),
             (0.93 ,1.79, 42.1),
             (0.82, 2.04, 43.5),
             (0.85, 2.25, 45.4),
             (0.74 ,2.43, 48.8)]

    node_x_target, node_y_target, time_target = nodes[1][0], nodes[1][1], (nodes[1][2] - nodes[0][2])*100
    robot_angle = np.arctan2((nodes[1][1]-nodes[0][1]),(nodes[1][0] - nodes[0][0]))
    robot_x, robot_y = nodes[0][0], nodes[0][1]
    case = "1"
    env = Gym_environment(renders=True, isDiscrete=False, motion="CW", case = case, init_angle = robot_angle, init_pos=[robot_x, robot_y])
    compensation_time = 0
    time_till_now=0
    predict_model =  prediction_model(case="1", node_x_target=node_x_target, node_y_target=node_y_target, robot_x=robot_x, robot_y=robot_y, robot_angle=robot_angle, time_target=time_target-compensation_time)

    # print(len(nodes))
    t0= time.time()
    for index in range(len(nodes)):
        node_x_target, node_y_target,time_target = nodes[index+1][0], nodes[index+1][1], 100*(nodes[index+1][2]-nodes[index][2])
        # predict_model =  prediction_model(self.case="1", node_x_target=node_x_target, node_y_target=node_y_target, robot_x=robot_x, robot_y=robot_y, robot_angle=robot_angle, time_target=time_target-compensation_time)
        start_time = time.time()
        predict_model.reset(case="1", node_x_target=node_x_target, node_y_target=node_y_target, robot_x=robot_x, robot_y=robot_y, robot_angle=robot_angle, time_target=time_target-compensation_time)
        model_description, compensation_time = predict_model.predict_model()
        # print(model_description, time_target, compensation_time)
        # print("prediction time: ", time.time()-start_time)
        # self.case = "1" #input("1: Ideal self.case, 2: Diff wheel radius, 3: Diff max wheel velocity: ")
        [MP_rotn, model_rotn, time_rotn, MP_lin, model_lin, time_lin] = model_description
        begin=time.time()
        if model_rotn!=0:
            robot_x, robot_y, robot_angle = predict_model.apply_model(env, MP_rotn, model_rotn, time_rotn, [robot_x, robot_y], robot_angle, start_time)
        if model_lin!=0:
            robot_x, robot_y, robot_angle = predict_model.apply_model(env, MP_lin, model_lin, time_lin, [robot_x, robot_y], robot_angle, start_time)
        # print("time to apply actions: ", time.time()-begin)
        time_till_now += time.time()-start_time
        print("Node: ", robot_x, robot_y, time.time()-t0)
if __name__ == '__main__':
  main()