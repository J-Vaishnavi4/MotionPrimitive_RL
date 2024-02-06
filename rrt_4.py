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

def calculate_time_bounds(dist, speed_limit=1.0):
    # Example dynamic time bounds calculation based on distance
    tmin = dist / (speed_limit * 100)  # Slightly faster scenario
    tmax = dist / (speed_limit * 1)  # Slightly slower scenario
    return tmin, tmax

def steer(from_node, to_node, extend_length, obstacle_list, speed_limit=1.0):
    dist = distance(from_node, to_node)
    if extend_length < dist:
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + extend_length * math.cos(theta)
        new_y = from_node.y + extend_length * math.sin(theta)
    else:
        new_x = to_node.x
        new_y = to_node.y
    tmin, tmax = calculate_time_bounds(extend_length, speed_limit)
    extend_time = np.random.uniform(tmin, tmax)
    new_node = Node(new_x, new_y, from_node.time + extend_time)
    new_node.cost = from_node.cost + extend_length + calculate_obstacle_proximity_penalty(new_node, obstacle_list)
    new_node.parent = from_node
    return new_node

def check_collision(from_node, to_node, obstacle_list):
    for (ox, oy, size) in obstacle_list:
        d = distance(Node(ox, oy), Node((from_node.x + to_node.x) / 2, (from_node.y + to_node.y) / 2))
        if d <= size:
            return True
    return False

def calculate_obstacle_proximity_penalty(node, obstacle_list):
    min_dist = min([distance(Node(ox, oy), node) - size for (ox, oy, size) in obstacle_list])
    penalty = 0
    if min_dist < 0.3:
        penalty = 100  # Arbitrary high penalty for being within 0.1 units of obstacles
    return penalty

def choose_parent(nn, new_node, nodes, r, obstacle_list):
    for node in nodes:
        if distance(node, new_node) < r and not check_collision(node, new_node, obstacle_list):
            potential_cost = node.cost + distance(node, new_node) + calculate_obstacle_proximity_penalty(new_node, obstacle_list)
            if potential_cost < new_node.cost:
                new_node.parent = node
                new_node.cost = potential_cost
    return new_node

def rewire(nodes, new_node, r, obstacle_list):
    for node in nodes:
        if node != new_node and distance(node, new_node) < r and not check_collision(new_node, node, obstacle_list):
            potential_cost = new_node.cost + distance(new_node, node) + calculate_obstacle_proximity_penalty(node, obstacle_list)
            if potential_cost < node.cost:
                node.parent = new_node
                node.cost = potential_cost
    return nodes

def generate_rrt_star(start, goal, obstacle_list, iter_max, area, r, max_dist, goal_bias, goal_radius, path_count, speed_limit=1.0):
    start.cost = 0.0  # Start node cost is zero
    nodes = [start]
    for _ in range(iter_max):
        if np.random.rand() > goal_bias:
            rnd_node = Node(np.random.uniform(area[0], area[1]), np.random.uniform(area[2], area[3]))
        else:
            rnd_node = goal
        nn = nearest_node(nodes, rnd_node)
        new_node = steer(nn, rnd_node, max_dist, obstacle_list, speed_limit)
        if not check_collision(nn, new_node, obstacle_list):
            nodes.append(choose_parent(nn, new_node, nodes, r, obstacle_list))
            nodes = rewire(nodes, new_node, r, obstacle_list)

    # Find nodes near the goal
    near_goal_nodes = [node for node in nodes if distance(node, goal) <= goal_radius]

    # Generate paths
    paths = []
    for end_node in near_goal_nodes:
        path = trace_path(end_node)
        paths.append(path)
        if len(paths) >= path_count:
            break

    return nodes, paths

def trace_path(node):
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    path.append(node)
    return path[::-1]

def plot_graph(nodes, paths, obstacle_list, start, goal):
    plt.figure(figsize=(10, 10))
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

# Example execution with the modified approach
start = Node(0, 0, 0)
goal = Node(5, 5, 0)
obstacle_list = [(2, 2, 0.5), (3, 3, 0.5), (4, 4, 0.5)]
iter_max = 1000
area = [-2, 7, -2, 7]
r = 1.0
max_dist = 0.75
goal_bias = 0.2
goal_radius = 0.5
path_count = 3
speed_limit = 1.0  # You can adjust this to simulate different movement speeds

nodes, paths = generate_rrt_star(start, goal, obstacle_list, iter_max, area, r, max_dist, goal_bias, goal_radius, path_count, speed_limit)
plot_graph(nodes, paths, obstacle_list, start, goal)
