import numpy as np
import math
import pylab as pl


class Node:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t
        self.cost = 0.0
        self.parent = None

def euclidean_distance(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2 + (node1.t - node2.t)**2)

def steer(from_node, to_node, min_dist, max_dist, t_min, t_max):
    """
    Steers from from_node towards to_node considering spatial and temporal constraints.
    """
    dist = euclidean_distance(from_node, to_node)
    if dist > max_dist:
        ratio = max_dist / dist
    elif dist < min_dist:
        ratio = min_dist / dist
    else:
        ratio = 1.0
    
    new_time = from_node.t + max(min((to_node.t - from_node.t) * ratio, t_max), t_min)
    # time_interval = ()
    new_x = from_node.x + (to_node.x - from_node.x) * ratio
    new_y = from_node.y + (to_node.y - from_node.y) * ratio
    
    return Node(new_x, new_y, new_time)

def custom_cost_function(from_node, to_node):
    """
    Define the custom cost function for moving from one node to another.
    """
    # Example cost function: Euclidean distance
    return euclidean_distance(from_node, to_node)

class RRTStar:
    def __init__(self, start, goal, x_bounds, y_bounds, t_bounds, min_dist, max_dist, t_min, t_max, max_iter=500):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.t_min = t_min
        self.t_max = t_max
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.t_bounds = t_bounds
        self.max_iter = max_iter
        self.nodes = [self.start]
    
    def sample(self):
        x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
        y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
        t = np.random.uniform(self.t_bounds[0], self.t_bounds[1])
        return Node(x, y, t)
    
    def nearest_node(self, sample):
        return min(self.nodes, key=lambda node: euclidean_distance(node, sample))
    
    def rewire(self, new_node):
        for node in self.nodes:
            if node == new_node or node == new_node.parent:
                continue
            if euclidean_distance(node, new_node) <= self.max_dist and new_node.cost + custom_cost_function(new_node, node) < node.cost:
                node.parent = new_node
                node.cost = new_node.cost + custom_cost_function(new_node, node)
    
    def find_path(self):
        for i in range(self.max_iter):
            sample = self.sample()
            nearest = self.nearest_node(sample)
            new_node = steer(nearest, sample, self.min_dist, self.max_dist, self.t_min, self.t_max)
            if euclidean_distance(new_node, nearest) <= self.max_dist:
                new_node.parent = nearest
                new_node.cost = nearest.cost + custom_cost_function(nearest, new_node)
                self.nodes.append(new_node)
                self.rewire(new_node)
        
        # Construct path
        path = []
        last_node = min(self.nodes, key=lambda node: euclidean_distance(node, self.goal))
        while last_node.parent is not None:
            path.append(last_node)
            last_node = last_node.parent
        path.append(self.start)
        return path[::-1]

# Example usage
start = (0, 0, 0)  # Start in the format (x, y, time)
goal = (10, 10, 20)  # Goal in the format (x, y, time)
x_bounds = (0, 15)
y_bounds = (0, 15)
t_bounds = (0, 1000)
min_dist = 0.5
max_dist = 2.0
t_min = 1
t_max = 5

rrt_star = RRTStar(start, goal, x_bounds, y_bounds, t_bounds, min_dist, max_dist, t_min, t_max)
path = rrt_star.find_path()

# Print path
path_x, path_y, path_t=[],[],[]
for node in path:
    print(f"({node.x}, {node.y}, {node.t})")
    path_x.append(node.x)
    path_y.append(node.y)
    path_t.append(node.t)
pl.plot(path_x,path_y)
for index in range(len(path_t)):
    pl.text(path_x[index]+np.random.uniform(-0.001,0.001),path_y[index]+np.random.uniform(-0.001,0.001),str(round(path_t[index],2)))
pl.show()