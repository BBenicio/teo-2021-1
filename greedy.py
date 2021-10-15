import numpy as np
from numba import njit, types

@njit(types.int64(types.int32[:,::1], types.int32[::1], types.int64, types.boolean[::1], types.int64, types.int64, types.float64))
def get_next_node(D, demands, Q, visited, current_node, current_capacity=0, alpha=0):
    candidate_list = (~visited & (current_capacity + demands < Q)).astype(np.bool8)
    if np.any(candidate_list):
        c_min = np.min(D[current_node][candidate_list])
        c_max = np.max(D[current_node][candidate_list])
        restricted_candidate_list = candidate_list & (D[current_node] <= c_min + alpha * (c_max - c_min))
        if np.any(restricted_candidate_list):
            rcl = np.argwhere(restricted_candidate_list).flatten()
            return np.random.choice(rcl)
        
    return -1

@njit(types.Tuple((types.int32[::1], types.boolean[::1]))(types.int32[:,::1], types.int32[::1], types.int64, types.float64))
def greedy(D, demands, Q, alpha=0):
    visited = np.full_like(demands, False, np.bool8)
    visited[0] = True
    route_start = np.full_like(demands, False, np.bool8)
    route = np.zeros_like(demands)
    i = 1
    while not np.all(visited):
        current_node = 0
        current_capacity = 0
        next_node = 0
        route_start[i] = True
        while next_node != -1:
            next_node = get_next_node(D, demands, Q, visited, current_node, current_capacity, alpha)
            if next_node != -1:
                current_capacity += demands[next_node]
                visited[next_node] = True
                route[i] = next_node
                current_node = next_node
                i += 1
    
    return route, route_start
