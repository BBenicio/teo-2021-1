import numpy as np
from numba import njit, types

@njit((types.int32[::1], types.boolean[::1]))
def swap(route, start):
    neighbourhood = []
    for i in range(1, route.shape[0]):
        for j in range(i+1, route.shape[0]):
            new_route = route.copy()
            new_route[i], new_route[j] = new_route[j], new_route[i]
            
            neighbourhood.append((new_route, start))
    
    return neighbourhood


@njit((types.int32[::1], types.boolean[::1]))
def two_opt(route, start):
    neighbourhood = []
    route_starts = np.argwhere(start).flatten()
    for p in range(route_starts.shape[0]):
        rs = route_starts[p]
        re = route_starts[p+1] if p+1 < route_starts.shape[0] else route.shape[0]
        
        for i in range(rs+1, re):
            for j in range(i+1, re):
                new_route = route.copy()
                new_route[i:j] = np.flip(new_route[i:j])

                neighbourhood.append((new_route, start))
    
    return neighbourhood


@njit([(types.int32[::1], types.int64, types.int64), (types.boolean[::1], types.int64, types.int64)])
def put_and_shift(array, i, j):
    new = np.zeros_like(array)
    new[j] = array[i]
    a, b = min(i, j), max(i, j)
    new[:a] = array[:a]
    new[b+1:] = array[b+1:]
    if i < j:
        new[i:j] = array[i+1:j+1]
    else:
        new[j+1:i+1] = array[j:i]

    return new


@njit((types.int32[::1], types.boolean[::1]))
def relocate(route, start):
    neighbourhood = []

    for i in range(1, route.shape[0]):
        for j in range(1, route.shape[0]):
            if i != j:
                new_route = put_and_shift(route, i, j)
                new_start = put_and_shift(start, i, j)
                neighbourhood.append((new_route, new_start))
    
    return neighbourhood


@njit((types.int32[::1], types.boolean[::1]))
def aggregate(route, start):
    return swap(route, start) + two_opt(route, start)
    # return relocate(route, start) + swap(route, start) + two_opt(route, start)


# TABU variants

@njit((types.int32[::1], types.boolean[::1]))
def tabu_swap(route, start):
    neighbourhood = []
    for i in range(1, route.shape[0]):
        for j in range(i+1, route.shape[0]):
            new_route = route.copy()
            new_route[i], new_route[j] = new_route[j], new_route[i]
            neighbourhood.append((new_route, start, (i, j)))
    
    return neighbourhood


@njit((types.int32[::1], types.boolean[::1]))
def tabu_two_opt(route, start):
    neighbourhood = []
    route_starts = np.argwhere(start).flatten()
    for p in range(route_starts.shape[0]):
        rs = route_starts[p]
        re = route_starts[p+1] if p+1 < route_starts.shape[0] else route.shape[0]
        
        for i in range(rs+1, re):
            for j in range(i+1, re):
                new_route = route.copy()
                new_route[i:j] = np.flip(new_route[i:j])

                neighbourhood.append((new_route, start, (i, j)))
    
    return neighbourhood