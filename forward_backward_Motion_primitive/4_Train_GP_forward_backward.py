from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared
from sklearn.model_selection import train_test_split

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

currentdir = os.path.dirname(__file__)

MP_name = input("Forward(F)/ Backward (B) MP: ")
if MP_name == "F":
    MP_name = "Forward"
    pass
elif  MP_name == "B":
    MP_name = "Backward"
else:
    raise SystemExit("Incorrect MP name")

if not os.path.exists("./GP_models/"+MP_name):
   os.makedirs("./GP_models/"+MP_name)

rows1 = pd.read_csv(os.path.join(currentdir, "./Samples/"+MP_name+"/samples_2_1.csv"), usecols=["displacement"])
rows2 = pd.read_csv(os.path.join(currentdir, "./Samples/"+MP_name+"/samples_2_1.csv"), usecols=["time"])

X = rows1.to_numpy()    # input to GP - Displacement (in metres)
y = rows2.to_numpy()    # output of GP - timesteps for which policy should be applied for the required displacement

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"-------------Gaussian process regression on noise-free dataset-------------------"

kernel = 1 * ExpSineSquared(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

plt.plot(X, y, label=r"$f(x) = timesteps$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label="Mean prediction")
# plt.fill_between(
#     X.ravel(),
#     mean_prediction - 1.96 * std_prediction,
#     mean_prediction + 1.96 * std_prediction,
#     alpha=0.5,
#     label=r"95% confidence interval",
# )
# plt.legend()
# plt.xlabel("$displacement$")
# plt.ylabel("$timestep$")
# _ = plt.title("Gaussian process regression on noise-free dataset")
plt.show()

with open("./GP_models/"+MP_name+"/no_noise_exp_2_1.dump" , "wb") as f:
     pickle.dump(gaussian_process, f)

model1 = pickle.load(open("./GP_models/"+MP_name+"/no_noise_exp_2_1.dump","rb"))


"-------------------Gaussian process regression on noisy dataset---------------------------"
rng = np.random.RandomState(1)
noise_std = 0.75
y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

gaussian_process = GaussianProcessRegressor(
    kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9
)
gaussian_process.fit(X_train, y_train_noisy)
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

plt.plot(X, y, label=r"$f(x) = timesteps$", linestyle="dotted")
plt.scatter(X_train, y_train_noisy, label = "Observations")

plt.plot(X, mean_prediction, label="Mean prediction")
# plt.fill_between(
#     X.ravel(),
#     mean_prediction - 1.96 * std_prediction,
#     mean_prediction + 1.96 * std_prediction,
#     color="tab:orange",
#     alpha=0.5,
#     label=r"95% confidence interval",
# )
# plt.legend()
# plt.xlabel("$displacement$")
# plt.ylabel("$timestep$")
# _ = plt.title("Gaussian process regression on a noisy dataset")

plt.show()

with open("./GP_models/"+MP_name+"/noisy_exp_2_1.dump" , "wb") as f:
     pickle.dump(gaussian_process, f)

model2 = pickle.load(open("./GP_models/"+MP_name+"/noisy_exp_2_1.dump","rb"))

"---------------PREDICTION------------"
required_displacement = 2.5  #metres
mean_prediction, std_prediction = model2.predict(np.array([required_displacement]).reshape(1, -1), return_std=True)
print("orientation change: "+str(required_displacement)+" rad \ntimesteps: "+str(mean_prediction[0])+"\nstandard deviation: "+ str(std_prediction[0]))
