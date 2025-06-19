import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# from Gym_env_translation import translation_env
# from Gym_env_rotation import rotation_env
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


    def linear_time_predict(self, model_num_lin):
        
        GP_linear = pickle.load(open("./"+self.case+"/GP_models/"+self.MP_name+str(model_num_lin)+"_noise.dump", "rb"))
        lin_predict_time, lin_std_prediction = GP_linear.predict(np.array([self.required_displacement]).reshape(1,-1), return_std=True)
        lin_predict_time = round(lin_predict_time[0][0])
        return lin_predict_time

    def predict_model(self):

        # self.case = "1" #input("1: Ideal case, 2: Diff wheel radius, 3: Diff max wheel velocity: ")

        if self.case == "1":
            self.case = "Ideal"
        elif self.case == "2":
            self.case = "Diff_wheel_radius"
        elif self.case == "3":
            self.case = "Diff_max_wheel_vel"
        else:
            raise SystemExit("Incorrect case number")
        
        rtn_model_num = [10,6,2]   
        lin_time_list = []


        if self.required_displacement>0 and len(lin_time_list)==0:
            self.MP_name = "F"
            MP_lin = 3
            lin_model_nums = [10,9,8,7,6,5,4,3,2]
            start_time_lin = time.time()
            with Pool(1) as p:
                # print(p.map(self.linear_time_predict,lin_model_nums))
                x= p.map(self.linear_time_predict,lin_model_nums)
            print("Time for lin model: ", time.time()-start_time_lin)
            lin_time_list = x
            

        if self.required_yaw_change==0:
            model_num_rotn = 0
            MP_rotn = 0
            rotn_predict_time = 0
            closest_time_cal =  min(lin_time_list, key=lambda x:abs(x-self.time_req))
            index = lin_time_list.index(closest_time_cal)
            lin_predict_time = lin_time_list[index]
            model_num_lin = lin_model_nums[index]
            total_predicted_time = lin_predict_time+rotn_predict_time
               
        else:
            if 0<self.required_yaw_change<=math.pi:
                MP_name = "CW"
                MP_rotn = 1
            else:
                MP_name = "CCW"
                MP_rotn = 2
                self.required_yaw_change = - self.required_yaw_change
            for model_num_rotn in rtn_model_num:
                GP_rotation = pickle.load(open("./"+self.case+"/GP_models/"+MP_name+str(model_num_rotn)+"_noise.dump", "rb"))
                rotn_predict_time, rotn_std_prediction = GP_rotation.predict(np.array([self.required_yaw_change]).reshape(1,-1), return_std=True)
                rotn_predict_time = round(rotn_predict_time[0][0])

                total_time_list = [x + rotn_predict_time for x in lin_time_list]
                closest_time_cal =  min(total_time_list, key=lambda x:abs(x-self.time_req))
                diff = abs(closest_time_cal-self.time_req)
                if diff<30:
                    index = total_time_list.index(closest_time_cal)
                    lin_predict_time = lin_time_list[index]
                    model_num_lin = lin_model_nums[index]
                    total_predicted_time = lin_predict_time+rotn_predict_time
                    break

        self.list_of_models.append([MP_rotn, model_num_rotn, rotn_predict_time, MP_lin, model_num_lin, lin_predict_time])
        compensation_time = total_predicted_time - self.time_req
        # print("times: ", total_predicted_time,self.time_req, compensation_time)
        # print("list of models: ", self.list_of_models[0])
        return self.list_of_models[0], compensation_time

def apply_model(env, MP_num, model_num, time_steps, pos_0, angle_0,start_time):
    case = "1"
    if MP_num == 1:
        MP_name = "CW"
        obs,info = env.reset_MP(motion = MP_name, init_angle = angle_0, init_position=[pos_0[0], pos_0[1]])
    elif MP_num == 2:
        MP_name = "CCW"
        obs,info = env.reset_MP(motion = MP_name, init_angle = angle_0, init_position=[pos_0[0], pos_0[1]])
    elif MP_num == 3:
        MP_name = "F"
        obs,info = env.reset_MP(motion = MP_name, init_angle = angle_0, init_position=[pos_0[0], pos_0[1]])
    # elif MP_num == 4:
    #     MP_name = "B"
    #     env = translation_env(renders=True, isDiscrete=False, motion="Backward", case = case,init_angle = angle_0, init_pos=[pos_0[0], pos_0[1]])
    if case =="1":
        case = "Ideal"
    model = ppo.PPO.load(os.path.join(currentdir,"./"+case+"/Policies/"+MP_name+"_"+str(model_num)))
    
    ld, x, y = [info['ld']], [info['x']], [info['y']]
    # init_time = time.time()
    for j in range(time_steps+1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done,truncated, info = env.step(action)
        ld.append(info['ld'])
        x.append(info['x'])
        y.append(info['y'])
        env.render(mode='human')
        # if MP_name =="F":
        # with open(os.path.join(currentdir,"./"+case)+'/new_traj.csv', 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([info['x'], info['y'], time.time()-start_time])
    
    return info['x'], info['y'], info['orientation']

def main():
    nodes = [(0.5, 0.5, 0),
    (0.6414213562373096, 0.6414213562373096, 5.471111777980823),
    (0.8313979816990338, 0.7039424069176664, 9.442909413535396),
    (0.9675247443543284, 0.8504671641738689, 13.147102090077595),
    (1.1036515070096231, 0.9969919214300714, 14.694445416654737),
    (1.2397782696649178, 1.143516678686274, 15.56531318507691),
    (1.3759050323202124, 1.2900414359424766, 16.467115088986162),
    (1.512031794975507, 1.436566193198679, 17.402109046549526),
    (1.6481585576308018, 1.5830909504548816, 18.292572749609132),
    (1.7842853202860964, 1.729615707711084, 19.409158662840863),
    (1.920412082941391, 1.8761404649672866, 20.79635135387038),
    (2.0565388455966858, 2.022665222223489, 21.60778670895039),
    (2.1926656082519806, 2.1691899794796914, 22.708422121939968),
    (2.3287923709072755, 2.315714736735894, 23.627634948227005),
    (2.46491913356257, 2.4622394939920964, 24.93196337727993)
    ]  
    node_x_target, node_y_target, time_target = nodes[1][0], nodes[1][1], (nodes[1][2] - nodes[0][2])*100
    robot_angle = np.arctan2((nodes[1][1]-nodes[0][1]),(nodes[1][0] - nodes[0][0]))
    robot_x, robot_y = nodes[0][0], nodes[0][1]
    case = "1"
    env = Gym_environment(renders=True, isDiscrete=False, motion="CW", case = case, init_angle = robot_angle, init_pos=[robot_x, robot_y])
    compensation_time = 0
    time_till_now=0
    # print(len(nodes))
    start_time = time.time()
    for index in range(len(nodes)):
        node_x_target, node_y_target,time_target = nodes[index+1][0], nodes[index+1][1], 100*(nodes[index+1][2]-nodes[index][2])
        predict_model =  prediction_model(case="1", node_x_target=node_x_target, node_y_target=node_y_target, robot_x=robot_x, robot_y=robot_y, robot_angle=robot_angle, time_target=time_target-compensation_time)

        model_description, compensation_time = predict_model.predict_model()
        # print(model_description, compensation_time)
        # case = "1" #input("1: Ideal case, 2: Diff wheel radius, 3: Diff max wheel velocity: ")
        [MP_rotn, model_rotn, time_rotn, MP_lin, model_lin, time_lin] = model_description
        
        if model_rotn!=0:
            robot_x, robot_y, robot_angle = apply_model(env, MP_rotn, model_rotn, time_rotn, [robot_x, robot_y], robot_angle, start_time)
        if model_lin!=0:
            robot_x, robot_y, robot_angle = apply_model(env, MP_lin, model_lin, time_lin, [robot_x, robot_y], robot_angle, start_time)
        time_till_now += time.time()-start_time
        print("Node: ", robot_x, robot_y, time.time()-start_time)
if __name__ == '__main__':
  main()