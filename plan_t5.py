import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

class Node:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t
        self.cost = 0.0
        self.parent = None

class Obstacle:
    def __init__(self, x_min, x_max, y_min, y_max, t_start, t_end):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.t_start = t_start
        self.t_end = t_end

    def collides(self, node):
        # Check if node's spatial position intersects with the obstacle's space
        spatial_collision = (self.x_min <= node.x <= self.x_max) and \
                            (self.y_min <= node.y <= self.y_max)
        # Check if node's time falls outside the obstacle's active time window
        temporal_collision = not (node.t < self.t_start or node.t > self.t_end)
        # A collision occurs only if both spatial and temporal conditions are met
        return spatial_collision and temporal_collision

def euclidean_distance(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2 + (node1.t - node2.t)**2)

def steer(from_node, to_node, min_dist, max_dist, t_min, t_max):
    dist = euclidean_distance(from_node, to_node)
    ratio = min(max_dist / dist, 1) if dist > min_dist else 1
    new_time = from_node.t + max(min((to_node.t - from_node.t) * ratio, t_max), t_min)
    new_x = from_node.x + (to_node.x - from_node.x) * ratio
    new_y = from_node.y + (to_node.y - from_node.y) * ratio
    return Node(new_x, new_y, new_time)

def custom_cost_function(from_node, to_node):
    

    return euclidean_distance(from_node, to_node)

class RRTStar:
    def __init__(self, start, goal, x_bounds, y_bounds, t_bounds, obstacles, min_dist, max_dist, t_min, t_max, max_iter, goal_bias, rewire_radius):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.t_min = t_min
        self.t_max = t_max
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.t_bounds = t_bounds
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius
        self.nodes = [self.start]

    def sample(self):
        if random.random() < self.goal_bias:
            return Node(self.goal.x, self.goal.y, self.goal.t)
        else:
            x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
            y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
            t = np.random.uniform(self.t_bounds[0], self.t_bounds[1])
            return Node(x, y, t)

    def nearest_node(self, sample):
        return min(self.nodes, key=lambda node: euclidean_distance(node, sample))

    def is_collision_free(self, node):
        for obstacle in self.obstacles:
            if obstacle.collides(node):
                return False
        return True

    def rewire(self, new_node):
        for node in self.nodes:
            if node == new_node or node == new_node.parent or not self.is_collision_free(node):
                continue
            if euclidean_distance(node, new_node) <= self.rewire_radius and \
               new_node.cost + custom_cost_function(new_node, node) < node.cost:
                node.parent = new_node
                node.cost = new_node.cost + custom_cost_function(new_node, node)

    def find_path(self):
        for i in range(self.max_iter):
            sample = self.sample()
            nearest = self.nearest_node(sample)
            new_node = steer(nearest, sample, self.min_dist, self.max_dist, self.t_min, self.t_max)
            if euclidean_distance(new_node, nearest) <= self.max_dist and self.is_collision_free(new_node):
                new_node.parent = nearest
                new_node.cost = nearest.cost + custom_cost_function(nearest, new_node)
                self.nodes.append(new_node)
                self.rewire(new_node)

        path = []
        last_node = min(self.nodes, key=lambda node: euclidean_distance(node, self.goal))
        while last_node.parent is not None:
            path.append(last_node)
            last_node = last_node.parent
        path.append(self.start)
        return path[::-1]

    def plot(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for node in self.nodes:
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], [node.t, node.parent.t], 'r-', alpha=0.4)

        for obstacle in self.obstacles:
            for z in np.linspace(obstacle.t_start, obstacle.t_end, 5):
                x = [obstacle.x_min, obstacle.x_max, obstacle.x_max, obstacle.x_min, obstacle.x_min]
                y = [obstacle.y_min, obstacle.y_min, obstacle.y_max, obstacle.y_max, obstacle.y_min]
                ax.plot(x, y, zs=z, zdir='z', color='black')

        for i in range(len(path) - 1):
            ax.plot([path[i].x, path[i+1].x], [path[i].y, path[i+1].y], [path[i].t, path[i+1].t], 'b-', linewidth=2)

        ax.scatter(self.start.x, self.start.y, self.start.t, c='green', label='Start')
        ax.scatter(self.goal.x, self.goal.y, self.goal.t, c='red', label='Goal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Time')
        plt.legend()
        plt.show()

# Define your parameters here
start = (0, 0, 0)
goal = (10, 10, 7)
x_bounds = (0, 15)
y_bounds = (0, 15)
t_bounds = (0, 20)
min_dist = 0.5
max_dist = 2.0
t_min = 1
t_max = 5
obstacles = [Obstacle(4, 6, 4, 6, 5, 7)]  # Temporal obstacle
max_iter = 10000
goal_bias = 0.2
rewire_radius = 2.0

rrt_star = RRTStar(start, goal, x_bounds, y_bounds, t_bounds, obstacles, min_dist, max_dist, t_min, t_max, max_iter, goal_bias, rewire_radius)
path = rrt_star.find_path()
rrt_star.plot(path)
