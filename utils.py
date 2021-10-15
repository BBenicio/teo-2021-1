import numpy as np
from numba import njit, types

def print_routes(route, start, D, demands):
    r = 0
    cost = 0
    prev = 0
    capacity = 0
    for i in range(1, route.shape[0]):
        if start[i]:
            if r > 0: print(f'\t({capacity})', end='')
            cost += D[prev, 0] # return to depot
            capacity = 0
            prev = 0
            r += 1
            print(f'\nRoute #{r}', end=': ')

        n = route[i]
        print(n, end=' ')
        capacity += demands[n]
        cost += D[prev, n]
        prev = n

    cost += D[prev, 0] # return to depot
    print(f'\t({capacity})')
    print(f'\nCost: {cost}')


@njit(types.boolean(types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def is_valid(route, start, demands, Q):
    current_capacity = 0
    visited = np.zeros_like(demands)
    visited[0] = 1
    if not start[1]: return False
    for i in range(1, route.shape[0]):
        visited[route[i]] += 1
        if start[i]:
            if current_capacity > Q:
                return False
            current_capacity = 0
        current_capacity += demands[route[i]]
    
    return np.all(visited == 1) and current_capacity <= Q


@njit(types.int64(types.int32[::1], types.boolean[::1], types.int32[:, ::1]))
def calculate_cost(route, start, D):
    cost = 0
    prev = 0
    for i in range(1, route.shape[0]):
        if start[i]:
            cost += D[prev, 0] # return to depot
            prev = 0
        n = route[i]
        cost += D[prev, n]
        prev = n
    cost += D[prev, 0] # return to depot
    return cost