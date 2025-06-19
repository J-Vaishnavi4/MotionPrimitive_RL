from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared
from sklearn.model_selection import train_test_split

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import csv
import pandas as pd

currentdir = os.path.dirname(__file__)

case = "1" #input("1: Ideal case, 2: Diff wheel radius, 3: Diff max wheel velocity: ")
algorithm = "1" #input("1: PPO, 2: SAC: ")
MP_name = input("1: CW, 2: CCW: 3: Forward, 4: Backward: ")
if MP_name == "1":
    MP_name = "CW"
    
elif MP_name == "2":
    MP_name = "CCW"
elif MP_name == "3":
    MP_name = "F"
elif MP_name == "4":
    MP_name = "B"

if case == "1":
    case = "Ideal"
elif case == "2":
    case = "Diff_wheel_radius"
elif case == "3":
    case = "Diff_max_wheel_vel"
else:
    raise SystemExit("Incorrect case number")

# if not os.path.exists("./"+case+"/GP_models"):
#     os.makedirs("./"+case+"/GP_models")

for model_num in range(3,4):
    rows1 = pd.read_csv("./"+case+"/Samples/"+MP_name+str(model_num)+'_samples.csv', usecols=["displacement"])
    rows2 = pd.read_csv("./"+case+"/Samples/"+MP_name+str(model_num)+'_samples.csv', usecols=["time"])
    rows1 = rows1.iloc[::50, :]
    rows2 = rows2.iloc[::50, :]
    model2 = pickle.load(open("./"+case+"/GP_models/"+MP_name+str(model_num)+"_noise.dump","rb"))
    X = rows1.to_numpy()    # input to GP - Displacement (in metres)
    y = rows2.to_numpy()    # output of GP - timesteps for which policy should be applied for the required displacement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(np.shape(X))
    "-------------Gaussian process regression on noise-free dataset-------------------"

    kernel = 1 * ExpSineSquared(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    # gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    # gaussian_process.fit(X_train, y_train)
    # gaussian_process.kernel_

    # mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
    # plt.plot(X, y, label=r"$f(x) = timesteps$", linestyle="dotted")
    # plt.scatter(X_train, y_train, label="Observations")
    # plt.plot(X, mean_prediction, label="Mean prediction")
    # # X = np.reshape(X,-1)
    # # mean_prediction = np.reshape(mean_prediction,-1)
    # # print(std_prediction)
    # # plt.fill_between(
    # #     X.ravel(),
    # #     mean_prediction - 19.6 * std_prediction,
    # #     mean_prediction + 19.6 * std_prediction,
    # #     alpha=0.5,
    # #     label=r"95% confidence interval",
    # # )
    # plt.legend()
    # plt.xlabel("$displacement$")
    # plt.ylabel("$timestep$")
    # _ = plt.title("Gaussian process regression on noise-free dataset")
    # plt.show()

    # with open("./"+case+"/GP_models/"+MP_name+str(model_num)+"_no_noise.dump" , "wb") as f:
    #     pickle.dump(gaussian_process, f)

    # model1 = pickle.load(open("./"+case+"/GP_models/"+MP_name+str(model_num)+"_no_noise.dump","rb"))

    "-------------------Gaussian process regression on noisy dataset---------------------------"
    rng = np.random.RandomState(1)
    noise_std = 2
    y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9
    )
    gaussian_process.fit(X_train, y_train_noisy)
    Xtest = np.arange(1,1*700,)/2000
    print(np.shape(Xtest))
    Xtest = Xtest.reshape(699,1)
    
    mean_prediction, std_prediction = model2.predict(Xtest, return_std=True)
    
    Xtest = np.reshape(Xtest,-1)
    mean_prediction = np.reshape(mean_prediction,-1)
    
    plt.plot(X, y, label=r"$f(x) = timesteps$", linestyle="dotted")
    plt.scatter(X_train, y_train_noisy, label = "Observations")

    plt.plot(Xtest, mean_prediction, label="Mean prediction")
    plt.fill_between(
        Xtest.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        color="tab:orange",
        alpha=0.5,
        label=r"95% confidence interval",
    )
    # plt.legend()
    plt.xlabel("$yaw change$")
    plt.ylabel("$timestep$")
    _ = plt.title("Gaussian process regression on noisy dataset")

    plt.show()
    mean_prediction=mean_prediction
    y=y
    # print("shape: ",np.shape(Xtest), np.shape(mean_prediction), np.shape(X.flatten()), np.shape(y.flatten()))
    # print(Xtest, mean_prediction, std_prediction)
    data = np.vstack((Xtest,mean_prediction/100, std_prediction)).T
    data1 = np.vstack((X.flatten(),y.flatten()/100)).T
    # np.savetxt('F_pred.csv', data, delimiter=',', header='Xtest, mean prediction', comments='')
    # np.savetxt('F_sample.csv', data1, delimiter=',', header='X,y', comments='')
    # with open(os.path.join(currentdir,"./"+case)+'/GP_sample_F.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([Xtest,mean_prediction,X,y])
    # model2 = pickle.load(open("./"+case+"/GP_models/"+MP_name+str(model_num)+"_noise.dump","rb"))
    # print(model2.get_params(deep=True))

    "---------------PREDICTION------------"
    required_displacement = 0.5  #metres
    mean_prediction, std_prediction = model2.predict(np.array([required_displacement]).reshape(1, -1), return_std=True)
    print("distance/orientation change: "+str(required_displacement)+" rad \ntimesteps: "+str(mean_prediction[0])+"\nstandard deviation: "+ str(std_prediction[0]))