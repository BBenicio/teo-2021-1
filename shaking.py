import numpy as np
from numba import njit, types
from greedy import get_next_node
from utils import is_valid
from operators import put_and_shift

@njit(types.Tuple((types.int32[::1], types.boolean[::1]))(types.int32[::1], types.boolean[::1], types.int32[:, ::1], types.int32[::1], types.int64, types.int64, types.float64))
def shake(route, start, D, demands, Q, k=10, alpha=0):
    remove_idx = np.sort(np.random.choice(np.arange(1, route.shape[0]), k, replace=False))
    visited = np.full_like(demands, True, dtype=np.bool8)
    
    new_route = route.copy()
    new_start = np.full_like(start, False)
    visited[route[remove_idx]] = False
    capacity = 0
    prev_idx = 0

    new_route[remove_idx] = -1
    new_start[1] = True

    for idx in remove_idx:
        # new_route[prev_idx:idx] = route[prev_idx:idx]
        # new_start[prev_idx:idx] = start[prev_idx:idx]
        for i in range(prev_idx+1, idx):
            if capacity + demands[new_route[i]] > Q:
                new_start[i] = True
            if new_start[i]:
                capacity = 0
            capacity += demands[new_route[i]]
        
        node = get_next_node(D, demands, Q, visited, new_route[idx-1], capacity, alpha)

        if node == -1:
            new_start[idx] = True
            capacity = 0
            node = get_next_node(D, demands, Q, visited, new_route[idx-1], capacity, alpha)

        visited[node] = True
        
        new_route[idx] = node
        
        if new_start[idx]:
            capacity = 0
        capacity += demands[node]

        prev_idx = idx
    
    for i in range(remove_idx[-1]+1, route.shape[0]):
        if capacity + demands[new_route[i]] > Q:
            new_start[i] = True
        if new_start[i]:
            capacity = 0
        capacity += demands[new_route[i]]

    return new_route, new_start


@njit((types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def rand_two_opt(route, start, demands, Q):
    route_starts = np.argwhere(start).flatten()
    rs, re = 0, 0
    while re - rs < 4:
        p = np.random.choice(route_starts)
        rs = route[p]
        re = route[p+1] if p+1 < route_starts.shape[0] else route.shape[0]
    
    irange = np.arange(rs+1, re)
    np.random.shuffle(irange)
    for i in irange:
        jrange = np.arange(i, re)
        np.random.shuffle(jrange)
        for j in jrange:
            if not start[i] and not start[j]:
                new_route = route.copy()
                new_route[rs+i:j] = np.flip(new_route[rs+i:j])
                if is_valid(new_route, start, demands, Q):
                    return new_route, start
        
    return route, start


@njit((types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def rand_relocate(route, start, demands, Q):
    irange = np.arange(1, route.shape[0])
    np.random.shuffle(irange)
    for i in irange:
        jrange = np.arange(i, route.shape[0])
        np.random.shuffle(jrange)
        for j in jrange:
            if i != j:
                new_route = put_and_shift(route, i, j)
                new_start = put_and_shift(start, i, j)
                if is_valid(new_route, new_start, demands, Q):
                    return new_route, new_start
    
    return route, start


@njit((types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def rand_swap(route, start, demands, Q):
    irange = np.arange(1, route.shape[0])
    np.random.shuffle(irange)
    for i in irange:
        jrange = np.arange(i, route.shape[0])
        np.random.shuffle(jrange)
        for j in jrange:
            if i != j:
                new_route = route.copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]
                if is_valid(new_route, start, demands, Q):
                    return new_route, start
    
    return route, start


@njit((types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def perturb(route, start, demands, Q):
    # route, start = rand_two_opt(route, start, demands, Q)
    route, start = rand_swap(route, start, demands, Q)
    # route, start = rand_relocate(route, start, demands, Q)
    return route, start