import numpy as np
from numba import njit
from greedy import greedy
from local_search import local_search
from operators import tabu_swap, tabu_two_opt
from shaking import shake, perturb
from utils import calculate_cost, is_valid

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
def simulated_annealing(route, start, D, demands, Q, T_max=5000, T_min=0.1, alpha=0.99, M=5.0, beta=1.05):
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

@njit
def tabu_search(route, start, D, demands, Q, T, Kmax):
    best_sol = (route, start)
    best_cost = calculate_cost(route, start, D)
    k = 0
    tabu_list = np.zeros_like(D, dtype=np.int32)

    next_cost = np.inf

    while k < Kmax:
        k += 1
        N = tabu_swap(route, start) + tabu_two_opt(route, start)
        movement = (0, 0)
        for n_route, n_start, m in N:
            if not is_valid(n_route, n_start, demands, Q): continue
            n_cost = calculate_cost(n_route, n_start, D)
            if (n_cost < next_cost and tabu_list[m] == 0) or n_cost < best_cost:
                next_sol = (n_route, n_start)
                next_cost = n_cost
                movement = m
                if n_cost < best_cost:
                    k = 0
                    best_sol = (n_route, n_start)
                    best_cost = n_cost
        route, start = next_sol

        for i in range(tabu_list.shape[0]):
            for j in range(tabu_list.shape[1]):
                if tabu_list[i,j] > 0:
                    tabu_list[i,j] -= 1
        tabu_list[movement] = T
    
    return best_sol

@njit
def do_tabu_search(route, start, D, demands, Q):
    n = D.shape[0]
    route, start = local_search(route, start, D, demands, Q)

    route, start = tabu_search(route, start, D, demands, Q, n // 3, 4 * n)
    route, start = tabu_search(route, start, D, demands, Q, n // 6, 2 * n)
    route, start = tabu_search(route, start, D, demands, Q, n**2 // 100, n)

    return route, start

# hybrid

@njit
def grasp_tabu(D, demands, Q, alpha=0.3, non_improving_iter=1000):
    best_cost = np.inf
    best_sol = None
    current_nii = 0
    while current_nii < non_improving_iter:
        route, start = greedy(D, demands, Q, alpha)
        route, start = tabu_search(route, start, D, demands, Q, D.shape[0] // 3, 4 * D.shape[0])
        cost = calculate_cost(route, start, D)
        if cost < best_cost:
            current_nii = 0
            best_cost = cost
            best_sol = (route, start)
        else:
            current_nii += 1

    return best_sol


@njit
def ils_tabu(route, start, D, demands, Q, k=10, alpha=0.3, non_improving_iter=1000):
    route, start = tabu_search(route, start, D, demands, Q, D.shape[0] // 3, 4 * D.shape[0])

    best_sol = (route, start)
    best_cost = calculate_cost(route, start, D)

    current_nii = 0
    while current_nii < non_improving_iter:
        route, start = shake(route, start, D, demands, Q, k, alpha)
        route, start = tabu_search(route, start, D, demands, Q, D.shape[0] // 3, 4 * D.shape[0])
        cost = calculate_cost(route, start, D)
        if cost < best_cost:
            current_nii = 0
            best_cost = cost
            best_sol = (route, start)
        else:
            current_nii += 1
    
    return best_sol