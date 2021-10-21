import numpy as np
from numba import njit
from greedy import greedy
from local_search import local_search
from shaking import shake, perturb
from utils import calculate_cost

@njit
def grasp(D, demands, Q, alpha=0.3, non_improving_iter=1000):
    best_cost = np.inf
    best_sol = None
    current_nii = 0
    while current_nii < non_improving_iter:
        route, start = greedy(D, demands, Q, alpha)
        route, start = local_search(route, start, D, demands, Q)
        cost = calculate_cost(route, start, D)
        if cost < best_cost:
            current_nii = 0
            best_cost = cost
            best_sol = (route, start)
        else:
            current_nii += 1

    return best_sol


@njit
def ils(route, start, D, demands, Q, k=10, alpha=0.3, non_improving_iter=1000):
    route, start = local_search(route, start, D, demands, Q)

    best_sol = (route, start)
    best_cost = calculate_cost(route, start, D)

    current_nii = 0
    while current_nii < non_improving_iter:
        route, start = shake(route, start, D, demands, Q, k, alpha)
        route, start = local_search(route, start, D, demands, Q)
        cost = calculate_cost(route, start, D)
        if cost < best_cost:
            current_nii = 0
            best_cost = cost
            best_sol = (route, start)
        else:
            current_nii += 1
    
    return best_sol


@njit
def simulated_annealing(route, start, D, demands, Q, T_max=5000, T_min=0.1, alpha=0.99, M=5, beta=1.05):
    cost = calculate_cost(route, start, D)

    best_sol = (route, start)
    best_cost = cost

    T = T_max
    while T > T_min:
        i = M
        while i >= 0:
            new_route, new_start = perturb(route, start, demands, Q)
            new_cost = calculate_cost(new_route, new_start, D)
            deltaE = new_cost - cost
            if deltaE < 0:
                route, start, cost = new_route, new_start, new_cost
                if cost < best_cost:
                    best_sol = (route, start)
            elif np.random.random() < np.exp(-deltaE / T):
                route, start, cost = new_route, new_start, new_cost
            
            i -= 1
        T *= alpha
        M *= beta
    
    return best_sol
