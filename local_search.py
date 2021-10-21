from numba import njit, types
from operators import aggregate
from utils import calculate_cost, is_valid

@njit(types.Tuple((types.int32[::1], types.boolean[::1]))(types.int32[::1], types.boolean[::1], types.int32[:, ::1], types.int32[::1], types.int64))
def local_search(route, start, D, demands, Q):
    best_cost = calculate_cost(route, start, D)
    best_sol = (route, start)
    cost = best_cost
    improved = True
    while improved:
        improved = False
        neighbourhood = aggregate(route, start)
        for r, s in neighbourhood:
            if is_valid(r, s, demands, Q):
                cost = calculate_cost(r, s, D)
                if cost < best_cost:
                    improved = True
                    best_cost = cost
                    best_sol = (r, s)
    
    return best_sol