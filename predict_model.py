import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# from Gym_env_translation import translation_env
# from Gym_env_rotation import rotation_env
from Gym_env import Gym_environment

import datetime
from stable_baselines3 import ppo, SAC
from stable_baselines3.common.env_checker import check_env
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math, time, csv

def predict_model():

    case = "1" #input("1: Ideal case, 2: Diff wheel radius, 3: Diff max wheel velocity: ")

    if case == "1":
        case = "Ideal"
    elif case == "2":
        case = "Diff_wheel_radius"
    elif case == "3":
        case = "Diff_max_wheel_vel"
    else:
        raise SystemExit("Incorrect case number")
    
    
    nodes = [(1.2263752211479197, 0.8813624063110713, 6.723624660997579),
        (1.4254042123793047, 0.8616783861572188, 7.566026561785433),
        (1.6214715751437427, 0.8222120807993785, 8.53973523218665),
        (1.8089644454451497, 0.7525958213039141, 12.023108605589552),
        (1.9972604889325283, 0.8200094698255542, 13.145694306282113),
        (2.185556532419907, 0.8874231183471943, 14.208243130271752),
        (2.3738525759072853, 0.9548367668688346, 15.03320903272038)
        ]  
    # nodes = [(0.5, 1.5, 0),(1.499, 1.513, 10.19),(2.497, 1.588, 15.85)]
    # nodes = [(0.5,1,0),(1.36, 1, 10.95),(2.06, 1, 15.46),(2.92,1,22.58),(3.5,1,26.33)]
    
    path_slope = np.arctan2((nodes[1][1]-nodes[0][1]),(nodes[1][0] - nodes[0][0]))
    # env = Gym_environment(renders=True, isDiscrete=False, motion="CW", case = case, init_angle = path_slope, init_pos=[nodes[0][0], nodes[0][1]])
    total_predicted_time = 0
    init_angle = path_slope
    list_of_models = []
    for pts in range(len(nodes)-1):
        print("points: ",pts)
        node_old, node_new= nodes[pts], nodes[pts+1]
        time_req = (node_new[2] - node_old[2])*100
        required_displacement = np.linalg.norm([node_new[0]-node_old[0], node_new[1]-node_old[1]])
        path_slope = np.arctan2((node_new[1]-node_old[1]),(node_new[0] - node_old[0]))
        required_yaw_change = - path_slope + init_angle
        # print("time req: ", time_req )
        if pts == 0:
            rotn_predict_time = 0
        model_num_rotn = 1
        print("req changes: ", required_displacement, required_yaw_change, time_req)
        rtn_model_num = [1]   
        for model_num_rotn in rtn_model_num:

            if required_yaw_change==0:
                model_num_rotn = 0
                MP_rotn = 0
                rotn_predict_time = 0
                # print("no rotation required")
                
            else:
                if 0< required_yaw_change <= math.pi:
                    MP_name = "CW"
                    MP_rotn = 1
                else:
                    MP_name = "CCW"
                    MP_rotn = 2
                    required_yaw_change = - required_yaw_change
                GP_rotation = pickle.load(open("./"+case+"/GP_models/"+MP_name+str(model_num_rotn)+"_noise.dump", "rb"))
                rotn_predict_time, rotn_std_prediction = GP_rotation.predict(np.array([required_yaw_change]).reshape(1,-1), return_std=True)
                rotn_predict_time = round(rotn_predict_time[0][0])
                init_angle = path_slope
                if rotn_predict_time > time_req:
                    continue 

            if required_displacement>0:
                MP_name = "F"
                MP_lin = 3
                for model_num_lin in range(1, 6):
                    
                    GP_linear = pickle.load(open("./"+case+"/GP_models/"+MP_name+str(model_num_lin)+"_noise.dump", "rb"))
                    lin_predict_time, lin_std_prediction = GP_linear.predict(np.array([required_displacement]).reshape(1,-1), return_std=True)
                    lin_predict_time = round(lin_predict_time[0][0])
                    total_predicted_time = rotn_predict_time + lin_predict_time
                    # print("AAA:",model_num_lin, total_predicted_time)
                    if (abs(time_req - total_predicted_time)<30):
                        break
            
            print(pts, "Prediction times: ",rotn_predict_time, lin_predict_time, total_predicted_time)
            print("rotn model num: ", model_num_rotn, "lin model num: ", model_num_lin)
            
            if (abs(time_req - total_predicted_time)<30):
                break
        list_of_models.append([pts, MP_rotn, model_num_rotn, rotn_predict_time, MP_lin, model_num_lin, lin_predict_time])

    print("LIST: ", total_predicted_time)
    # print(len(list_of_models))
    return list_of_models, nodes

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
    init_time = time.time()
    for j in range(time_steps+1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done,truncated, info = env.step(action)
        ld.append(info['ld'])
        x.append(info['x'])
        y.append(info['y'])
        env.render(mode='human')
        # with open(os.path.join(currentdir,"./"+case)+'/trajectory_11.csv', 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([info['x'], info['y'], time.time()-start_time])
    
    return info['x'], info['y'], info['orientation']

def main():
    model_description, nodes = predict_model()
    print(model_description)
    case = "1" #input("1: Ideal case, 2: Diff wheel radius, 3: Diff max wheel velocity: ")
    

    x_new, y_new, new_orientation = nodes[0][0], nodes[0][1], np.arctan2((nodes[1][1]-nodes[0][1]),(nodes[1][0] - nodes[0][0]))
    # print("x_new", x_new)
    env = Gym_environment(renders=True, isDiscrete=False, motion="CW", case = case, init_angle = new_orientation, init_pos=[x_new, y_new])
    obs,info = env.reset()
    start_time = time.time()

    for index in range(len(model_description)):
        # print("Node: ", x_new, y_new)
        [node_num, MP_rotn, model_rotn, time_rotn, MP_lin, model_lin, time_lin] = model_description[index]
        if model_rotn!=0:
            x_new, y_new, new_orientation = apply_model(env, MP_rotn, model_rotn, time_rotn, [x_new, y_new], new_orientation, start_time)
        if model_lin!=0:
            x_new, y_new, new_orientation = apply_model(env, MP_lin, model_lin, time_lin, [x_new, y_new], new_orientation, start_time)
        print("Node: ", x_new, y_new)
if __name__ == '__main__':
  main()