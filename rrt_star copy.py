import numpy as np
import matplotlib.pyplot as plt
import math

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None

def distance(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def nearest_node(nodes, n):
    return min(nodes, key=lambda node: distance(node, n))

def steer(from_node, to_node, extend_length=float('inf')):
    dist = distance(from_node, to_node)
    if extend_length > dist:
        extend_length = dist
    theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
    new_node = Node(from_node.x + extend_length * math.cos(theta),
                    from_node.y + extend_length * math.sin(theta))
    new_node.cost = from_node.cost + extend_length
    new_node.parent = from_node
    return new_node

def choose_parent(nn, new_node, nodes, r, obstacle_list):
    for node in nodes:
        if distance(node, new_node) < r and node.cost + distance(node, new_node) < new_node.cost and not check_collision(node, new_node, obstacle_list):
            new_node.parent = node
            new_node.cost = node.cost + distance(node, new_node)
    return new_node

def rewire(nodes, new_node, r, obstacle_list):
    for node in nodes:
        if node != new_node and distance(node, new_node) < r and new_node.cost + distance(new_node, node) < node.cost and not check_collision(new_node, node, obstacle_list):
            node.parent = new_node
            node.cost = new_node.cost + distance(new_node, node)
    return nodes

def check_collision(from_node, to_node, obstacle_list):
    for (ox, oy, size) in obstacle_list:
        d = distance(Node(ox, oy), Node((from_node.x + to_node.x) / 2, (from_node.y + to_node.y) / 2))
        if d <= size + math.sqrt(((from_node.x - to_node.x) / 2)**2 + ((from_node.y - to_node.y) / 2)**2):
            return True
    return False

def find_near_goal_nodes(nodes, goal, radius):
    return [node for node in nodes if distance(node, goal) <= radius]

def trace_path(node):
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    path.append(node)
    return path[::-1]

def generate_rrt_star(start, goal, obstacle_list, iter_max, area, r, goal_radius=0.5, path_count=50):
    nodes = [start]
    for _ in range(iter_max):
        rnd_node = Node(np.random.uniform(area[0], area[1]), np.random.uniform(area[2], area[3]))
        nn = nearest_node(nodes, rnd_node)
        new_node = steer(nn, rnd_node, extend_length=1.5)
        if not check_collision(nn, new_node, obstacle_list):
            nodes.append(choose_parent(nn, new_node, nodes, r, obstacle_list))
            nodes = rewire(nodes, new_node, r, obstacle_list)
    
    # Find nodes near the goal
    near_goal_nodes = find_near_goal_nodes(nodes, goal, goal_radius)
    
    # Generate paths
    paths = []
    for end_node in near_goal_nodes:
        if len(paths) < path_count:
            path = trace_path(end_node)
            paths.append(path)
        else:
            break
    
    return nodes, paths

def plot_graph(nodes, paths, obstacle_list, start, goal):
    plt.figure()
    plt.plot(start.x, start.y, "xr")
    plt.plot(goal.x, goal.y, "xg")
    for node in nodes:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")
    for (ox, oy, size) in obstacle_list:
        circle = plt.Circle((ox, oy), size, color='black', fill=False)
        plt.gca().add_artist(circle)
    for path in paths:
        plt.plot([node.x for node in path], [node.y for node in path], '-r')
    plt.grid(True)
    plt.axis("equal")
    plt.show()

# Parameters
start = Node(0, 0)
goal = Node(5, 5)
obstacle_list = [(2, 2, 0.5), (3, 3, 0.5), (4, 4, 0.5)]
iter_max = 5000
area = [-2, 7, -2, 7]
r = 1.0  # Radius for choosing parent and rewire operations
goal_radius = 0.5  # Radius within which nodes are considered close to the goal
path_count = 50  # Number of paths to find

nodes, paths = generate_rrt_star(start, goal, obstacle_list, iter_max, area, r, goal_radius, path_count)
plot_graph(nodes, paths, obstacle_list, start, goal)
