import numpy as np
from math import sqrt


def read_meta_section(f):
    line = f.readline()
    metadata = {}
    while 'NODE_COORD_SECTION' not in line:
        split = [ w.strip() for w in line.split(' : ') ]
        key, value = split[0].lower(), split[1]
        metadata[key] = value
        line = f.readline()
    metadata['dimension'] = int(metadata['dimension'])
    metadata['capacity'] = int(metadata['capacity'])
    return metadata


def read_coord_section(f, dimension):
    line = f.readline()
    nodes = [{ 'x': None, 'y': None, 'demand': None } for i in range(dimension)]
    while 'DEMAND_SECTION' not in line:
        node, x, y = [ int(w.strip()) for w in line.strip().split(' ') ]
        nodes[node - 1]['x'] = x
        nodes[node - 1]['y'] = y
        line = f.readline()
    
    return nodes


def read_demand_section(f, nodes):
    line = f.readline()
    while 'DEPOT_SECTION' not in line:
        node, demand = [ int(w.strip()) for w in line.strip().split(' ') ]
        nodes[node - 1]['demand'] = demand
        line = f.readline()

    return nodes


def read_depot_section(f):
    line = f.readline()
    depots = []
    while '-1' not in line:
        node = int(line.strip())
        if line[node - 1]:
            depots.append(node)
        line = f.readline()
    
    return depots


def read_cvrp(file_path):
    meta = None
    nodes = None
    depots = None
    with open(file_path, 'r') as f:
        meta = read_meta_section(f)
        nodes = read_coord_section(f, meta['dimension'])
        nodes = read_demand_section(f, nodes)
        depots = read_depot_section(f)
    
    return meta, nodes, depots


def dist(node_a, node_b):
    return round(sqrt((node_a['x'] - node_b['x']) ** 2 + (node_a['y'] - node_b['y']) ** 2))


def get_distance_matrix(meta, nodes):
    distance_matrix = np.zeros((meta['dimension'], meta['dimension']), dtype=int)
    for i in range(distance_matrix.shape[0]):
        for j in range(i+1, distance_matrix.shape[1]):
            distance_matrix[i, j] = distance_matrix[j, i] = dist(nodes[i], nodes[j])
    
    return distance_matrix


def prepare_input(filepath):
    meta, nodes, _ = read_cvrp(filepath)
    D = get_distance_matrix(meta, nodes)
    demands = np.array([n['demand'] for n in nodes])
    Q = meta['capacity']

    return D, demands, Q
    