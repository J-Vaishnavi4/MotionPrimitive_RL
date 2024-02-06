import numpy as np
import matplotlib.pyplot as plt
import math

class Node:
    def __init__(self, x, y, time=0.0):
        self.x = x
        self.y = y
        self.time = time  # Time to reach this node
        self.cost = float('inf')  # Initialize cost as infinite
        self.parent = None

def distance(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def nearest_node(nodes, n):
    return min(nodes, key=lambda node: distance(node, n))

def steer(from_node, to_node, tmin, tmax, extend_length):
    dist = distance(from_node, to_node)
    if extend_length < dist:
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + extend_length * math.cos(theta)
        new_y = from_node.y + extend_length * math.sin(theta)
    else:
        new_x = to_node.x
        new_y = to_node.y
    extend_time = np.random.uniform(tmin, tmax)  # Sample time to travel
    new_node = Node(new_x, new_y, from_node.time + extend_time)
    new_node.cost = from_node.cost + extend_length  # Initially set cost based on distance
    new_node.parent = from_node
    return new_node

def check_collision(from_node, to_node, obstacle_list):
    for (ox, oy, size) in obstacle_list:
        d = distance(Node(ox, oy), Node((from_node.x + to_node.x) / 2, (from_node.y + to_node.y) / 2))
        if d <= size:
            return True
    return False

def calculate_obstacle_proximity_penalty(node, obstacle_list):
    min_dist = min(distance(Node(ox, oy), node) - size for (ox, oy, size) in obstacle_list)
    penalty = 0
    if min_dist < 0.3:
        penalty = 1000  # Arbitrary high penalty for being within 0.1 units of obstacles
    return penalty

def choose_parent(nn, new_node, nodes, r, obstacle_list):
    for node in nodes:
        if distance(node, new_node) < r and node.cost + distance(node, new_node) + calculate_obstacle_proximity_penalty(new_node, obstacle_list) < new_node.cost and not check_collision(node, new_node, obstacle_list):
            new_node.parent = node
            new_node.cost = node.cost + distance(node, new_node) + calculate_obstacle_proximity_penalty(new_node, obstacle_list)
            new_node.time = node.time + new_node.time  # Update time based on parent
    return new_node

def rewire(nodes, new_node, r, obstacle_list):
    for node in nodes:
        if node != new_node and distance(node, new_node) < r and new_node.cost + distance(new_node, node) + calculate_obstacle_proximity_penalty(node, obstacle_list) < node.cost and not check_collision(new_node, node, obstacle_list):
            node.parent = new_node
            node.cost = new_node.cost + distance(new_node, node) + calculate_obstacle_proximity_penalty(node, obstacle_list)
            node.time = new_node.time + node.time  # Update time based on new parent
    return nodes

def generate_rrt_star(start, goal, obstacle_list, iter_max, area, r, tmin, tmax, max_dist, goal_bias, goal_radius=0.5, path_count=50):
    start.cost = 0.0  # Start node cost is zero
    nodes = [start]
    for _ in range(iter_max):
        if np.random.rand() > goal_bias:  # Apply goal bias
            rnd_node = Node(np.random.uniform(area[0], area[1]), np.random.uniform(area[2], area[3]))
        else:
            rnd_node = goal  # Directly sample the goal position
        nn = nearest_node(nodes, rnd_node)
        new_node = steer(nn, rnd_node, tmin, tmax, extend_length=max_dist)
        if not check_collision(nn, new_node, obstacle_list):
            nodes.append(choose_parent(nn, new_node, nodes, r, obstacle_list))
            nodes = rewire(nodes, new_node, r, obstacle_list)
    
    # Find nodes near the goal
    near_goal_nodes = find_near_goal_nodes(nodes, goal, goal_radius)
    
    # Generate paths
    paths = []
    for end_node in near_goal_nodes:
        path = trace_path(end_node)
        paths.append(path)
        if len(paths) >= path_count:
            break
    
    return nodes, paths

def find_near_goal_nodes(nodes, goal, radius):
    return [node for node in nodes if distance(node, goal) <= radius]

def trace_path(node):
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    path.append(node)
    return path[::-1]

def plot_graph(nodes, paths, obstacle_list, start, goal):
    plt.figure()
    plt.plot(start.x, start.y, "xr", label="Start")
    plt.plot(goal.x, goal.y, "xg", label="Goal")
    for node in nodes:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-k")
    for (ox, oy, size) in obstacle_list:
        circle = plt.Circle((ox, oy), size, color='black', fill=True)
        plt.gca().add_artist(circle)
    for path in paths:
        plt.plot([node.x for node in path], [node.y for node in path], 'r-', linewidth=2)
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# Example parameters
start = Node(0, 0, 0)
goal = Node(5, 5, 0)
obstacle_list = [(2, 2, 0.5), (3, 3, 0.5), (4, 4, 0.5)]
iter_max = 5000
area = [-2, 7, -2, 7]
r = 1.0
tmin = 0.1
tmax = 0.5
max_dist = 0.75  # Max distance a node can extend towards a new node
goal_bias = 0.2  # Probability of sampling the goal position
goal_radius = 0.5
path_count = 10

nodes, paths = generate_rrt_star(start, goal, obstacle_list, iter_max, area, r, tmin, tmax, max_dist, goal_bias, goal_radius, path_count)
plot_graph(nodes, paths, obstacle_list, start, goal)
