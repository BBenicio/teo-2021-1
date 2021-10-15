import numpy as np
from numba import njit, types

from utils import is_valid

@njit((types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def swap(route, start, demands, Q):
    neighbourhood = []
    for i in range(1, route.shape[0]):
        for j in range(i+1, route.shape[0]):
            new_route = route.copy()
            new_start = start.copy()
            new_route[i], new_route[j] = new_route[j], new_route[i]
            new_start[i], new_start[j] = new_start[j], new_start[i]
            # if is_valid(new_route, new_start, demands, Q):
            neighbourhood.append((new_route, start))
    
    return neighbourhood


@njit((types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def two_opt(route, start, demands, Q):
    neighbourhood = []
    route_starts = np.argwhere(start).flatten()
    for p in range(route_starts.shape[0]):
        rs = route[p]
        re = route[p+1] if p+1 < route_starts.shape[0] else route.shape[0]
        
        for i in range(rs+1, re):
            for j in range(i+1, re):
                if not start[i] and not start[j]:
                    new_route = route.copy()
                    new_route[rs+i:j] = np.flip(new_route[rs+i:j])
                    # if is_valid(new_route, start, demands, Q):
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


@njit((types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def relocate(route, start, demands, Q):
    neighbourhood = []

    for i in range(1, route.shape[0]):
        for j in range(1, route.shape[0]):
            if i != j:
                new_route = put_and_shift(route, i, j)
                new_start = put_and_shift(start, i, j)
                # if is_valid(new_route, new_start, demands, Q):
                neighbourhood.append((new_route, new_start))
    
    return neighbourhood


@njit((types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def aggregate(route, start, demands, Q):
    return relocate(route, start, demands, Q) + swap(route, start, demands, Q) + two_opt(route, start, demands, Q)
