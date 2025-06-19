import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
# Volume dimensions
volume_length = 3
volume_width = 3
volume_height = 20
 
# Obstacle dimensions and coordinates
obstacle_base_coords = np.array([[1.2, 0, 0], [1.8, 0, 0], [1.8, 2.4, 0], [1.2, 2.4, 0]])
obstacle_height = 5
 
# Create vertices for the obstacle
vertices = [list(map(list, obstacle_base_coords))]
# Add the top vertices by adding the height to the z-coordinate
top_vertices = obstacle_base_coords + np.array([0, 0, obstacle_height])
vertices.append(list(map(list, top_vertices)))
 
# Create sides of the obstacle as polygons
polygons = [[vertices[0][i], vertices[0][(i+1)%4], vertices[1][(i+1)%4], vertices[1][i]] for i in range(4)]
# Add the top face
polygons.append(list(map(list, top_vertices)))
 
# Plot the obstacle
ax.add_collection3d(Poly3DCollection(polygons, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))
 
# Waypoints
waypoints = np.array([
(0.5, 1.0, 0),
(0.6838899657874129, 0.921358532041269, 2.979416312762666),
(0.8837027214554057, 0.9300108538315747, 4.706719552934777),
(1.0268073706890746, 0.8682218940213605, 5.523277443841472),
(1.2263752211479197, 0.8813624063110713, 6.723624660997579),
(1.4254042123793047, 0.8616783861572188, 7.566026561785433),
(1.6214715751437427, 0.8222120807993785, 8.53973523218665),
(1.8089644454451497, 0.7525958213039141, 12.023108605589552),
(1.9972604889325283, 0.8200094698255542, 13.145694306282113),
(2.185556532419907, 0.8874231183471943, 14.208243130271752),
(2.3738525759072853, 0.9548367668688346, 15.03320903272038)
])
 
# Plot waypoints
ax.scatter(waypoints[:,0], waypoints[:,1], waypoints[:,2], color='black', s=100)
 
# Connect waypoints with lines
for i in range(len(waypoints)-1):
    ax.plot([waypoints[i][0], waypoints[i+1][0]], [waypoints[i][1], waypoints[i+1][1]], [waypoints[i][2], waypoints[i+1][2]], 'k--')
 
# Setting the axes properties
ax.set_xlim([0, volume_length])
ax.set_ylim([0, volume_width])
ax.set_zlim([0, volume_height])
 
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
 
plt.title('3D Volume with Obstacle and Waypoints')
# plt.show()

data_2d = np.array([(0.5, 0.5, 0),
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
])

data_2d = np.array([(0.5 ,0.5,0),
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
            (0.74 ,2.43, 48.8)])

# nodes = np.array([(0.5,1,0),(1.36, 1, 10.95),(2.06, 1, 15.46),(2.92,1,22.58),(3.5,1,26.33)])
print(data_2d[:,0])
plt.figure()
case = "Ideal"
rows1 = pd.read_csv("./"+case+"/new_traj_two_target1.csv", usecols=["X"])
rows2 = pd.read_csv("./"+case+"/new_traj_two_target1.csv", usecols=["Y"])
print(rows1.to_numpy(), rows1)
plt.plot(rows1.to_numpy(), rows2.to_numpy() )
plt.plot(data_2d[:,0],data_2d[:,1],linestyle='dashed')
plt.xlim([-0.5,4])
plt.ylim([-0.5,4])
plt.grid()
plt.legend(['actual trajectory', 'path planner'])
plt.show()