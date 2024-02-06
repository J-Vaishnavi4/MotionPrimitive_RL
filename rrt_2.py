import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class Node:
    def __init__(self, point):
        self.point = point
        self.parent = None
        self.cost = 0.0

class Obstacle:
    def __init__(self, center, size, shape='rect'):
        self.center = center
        self.size = size  # For circles, size is radius; for rectangles, size is [length, width].
        self.shape = shape

def is_collision(node, new_node, obstacles):
    if new_node.point[0] < 0 or new_node.point[0] > 100 or new_node.point[1] < 0 or new_node.point[1] > 100:
        return True  # Assuming a 100x100 grid
    
    for obstacle in obstacles:
        if obstacle.shape == 'rect':
            if (obstacle.center[0] - obstacle.size[0]/2 <= new_node.point[0] <= obstacle.center[0] + obstacle.size[0]/2 and
                obstacle.center[1] - obstacle.size[1]/2 <= new_node.point[1] <= obstacle.center[1] + obstacle.size[1]/2):
                return True
        elif obstacle.shape == 'circle':
            if distance.euclidean(new_node.point, obstacle.center) <= obstacle.size:
                return True
    return False

def find_nearest(nodes, new_node):
    distances = [distance.euclidean(node.point, new_node.point) for node in nodes]
    nearest_index = distances.index(min(distances))
    return nodes[nearest_index]

def steer(from_node, to_point, max_dist):
    if distance.euclidean(from_node.point, to_point) < max_dist:
        return Node(to_point)
    else:
        theta = np.arctan2(to_point[1] - from_node.point[1], to_point[0] - from_node.point[0])
        return Node([from_node.point[0] + max_dist * np.cos(theta), from_node.point[1] + max_dist * np.sin(theta)])

def path_to_start(node):
    path = []
    while node.parent is not None:
        path.append(node.point)
        node = node.parent
    path.append(node.point)
    return path

def RRT_star(start, goal, obstacles, max_dist, num_paths, max_iterations):
    nodes = [Node(start)]
    paths = []
    
    for _ in range(max_iterations):
        rand_point = [np.random.uniform(0, 100), np.random.uniform(0, 100)]
        nearest_node = find_nearest(nodes, Node(rand_point))
        new_node = steer(nearest_node, rand_point, max_dist)
        
        if not is_collision(nearest_node, new_node, obstacles):
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + distance.euclidean(nearest_node.point, new_node.point)
            nodes.append(new_node)
            
            if distance.euclidean(new_node.point, goal) <= max_dist:
                goal_node = steer(new_node, goal, max_dist)
                if not is_collision(new_node, goal_node, obstacles):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + distance.euclidean(new_node.point, goal)
                    paths.append(path_to_start(goal_node))
                    if len(paths) == num_paths:
                        break
    
    # Optional: Sort paths by length or cost
    paths.sort(key=lambda x: len(x))
    return paths[:num_paths]

# Define obstacles
obstacles = [
    Obstacle(center=[50, 50], size=[10, 20], shape='rect'),  # Rectangular obstacle
    Obstacle(center=[30, 30], size=10, shape='circle'),  # Circular obstacle
]

# Parameters
start = [0, 0]
goal = [100, 100]
max_dist = 1
num_paths = 3
max_iterations = 100000

paths = RRT_star(start, goal, obstacles, max_dist, num_paths, max_iterations)

# Plotting
plt.figure(figsize=(10, 10))
for obstacle in obstacles:
    if obstacle.shape == 'rect':
        plt.gca().add_patch(plt.Rectangle((obstacle.center[0] - obstacle.size[0]/2, obstacle.center[1] - obstacle.size[1]/2), obstacle.size[0], obstacle.size[1], color='red'))
    elif obstacle.shape == 'circle':
        plt.gca().add_patch(plt.Circle(obstacle.center, obstacle.size, color='red'))

for path in paths:
    plt.plot([point[0] for point in path], [point[1] for point in path])

plt.plot(start[0], start[1], 'go')  # Start
plt.plot(goal[0], goal[1], 'bo')  # Goal
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('RRT* Paths')
plt.show()
