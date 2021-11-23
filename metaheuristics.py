import numpy as np
from numba import njit
from greedy import greedy
from local_search import local_search
from operators import tabu_swap, tabu_two_opt
from shaking import shake, perturb
from utils import calculate_cost, is_valid

@njit
def grasp(route, start, D, demands, Q, alpha=0.3, non_improving_iter=1000):
    '''Aplica uma heurística baseada no GRASP.

    A cada iteração é gerada uma solução de forma semi-gulosa (controlada pelo alpha)
    e realizada uma busca local a partir desta solução. A melhor solução
    encontrada é retornada se não houver melhora em k iterações.
    '''
    # recebe a solução inicial nos parâmetros, preciso executar BL
    route, start = local_search(route, start, D, demands, Q)

    # primeira e melhor solução encontrada
    best_cost = calculate_cost(route, start, D)
    best_sol = (route, start)
    
    # continua iterando até que não tenha havido melhora por muitas iterações
    # critério de parada
    current_nii = 0
    while current_nii < non_improving_iter:
        route, start = greedy(D, demands, Q, alpha)
        route, start = local_search(route, start, D, demands, Q)
        cost = calculate_cost(route, start, D)
        if cost < best_cost: # se for melhor que a melhor solução atual, atualiza
            current_nii = 0
            best_cost = cost
            best_sol = (route, start)
        else:
            current_nii += 1

    return best_sol


@njit
def ils(route, start, D, demands, Q, k=10, alpha=0.3, non_improving_iter=1000):
    '''Aplica uma heurística baseada no ILS.

    A cada iteração a solução encontrada é perturbada para gerar uma nova solução.
    Esta perturbação funciona removendo k vértices aleatórios e reconstruindo a
    solução.
    '''
    route, start = local_search(route, start, D, demands, Q)

    best_sol = (route, start)
    best_cost = calculate_cost(route, start, D)

    # critério de parada: muitas iterações sem melhora
    current_nii = 0
    while current_nii < non_improving_iter:
        # perturba a solução atual e executa BL sobre a solução perturbada
        route, start = shake(route, start, D, demands, Q, k, alpha)
        route, start = local_search(route, start, D, demands, Q)
        
        # aceita qualquer solução, seja melhor que a atual ou não

        cost = calculate_cost(route, start, D)
        if cost < best_cost: # armazena separadamente se for a melhor encontrada
            current_nii = 0
            best_cost = cost
            best_sol = (route, start)
        else:
            current_nii += 1
    
    return best_sol


@njit
def simulated_annealing(route, start, D, demands, Q, T_max=5000, T_min=0.1, alpha=0.99, M=5.0, beta=1.05):
    '''Aplica uma heurística baseada no Simulated Annealing para CVRP proposta
    por Harmanani et al. (2011).

    Em cada temperatura são executadas M iterações (M é atualizada conforme
    valor de beta). As soluções são perturbadas e aceitas com base na diferença
    de qualidade entre si e a solução gerada anteriormente.
    '''
    cost = calculate_cost(route, start, D)

    best_sol = (route, start)
    best_cost = cost

    # itera até que a temperatura atinja o mínimo
    T = T_max
    while T > T_min:
        # em uma temperatura iteramos M vezes
        i = M
        while i >= 0:
            new_route, new_start = perturb(route, start, demands, Q)
            new_cost = calculate_cost(new_route, new_start, D)
            deltaE = new_cost - cost
            
            # sempre aceita soluções melhores
            if deltaE < 0:
                route, start, cost = new_route, new_start, new_cost
                if cost < best_cost:
                    best_sol = (route, start)
            # soluções piores são aceitas com certa probabilidade
            elif np.random.random() < np.exp(-deltaE / T):
                route, start, cost = new_route, new_start, new_cost
            
            i -= 1
        # diminui a temperatura e aumenta M
        T *= alpha
        M *= beta
    
    return best_sol

@njit
def tabu_search(route, start, D, demands, Q, T, Kmax):
    '''Aplica uma heurística baseada na Busca Tabu, segundo a proposta de
    Oliveira et al. (2020).

    Nesta abordagem um movimento permanece na lista tabu por T iterações e
    o critério de parada são Kmax iterações sem melhora.
    '''
    best_sol = (route, start)
    best_cost = calculate_cost(route, start, D)
    k = 0
    tabu_list = np.zeros_like(D, dtype=np.int32)

    next_cost = np.inf

    while k < Kmax:
        k += 1
        # gera a vizinhança usando swap e 2-opt
        N = tabu_swap(route, start) + tabu_two_opt(route, start)
        movement = (0, 0)
        for n_route, n_start, m in N:
            # caso a solução vizinha seja inválida, descarte
            if not is_valid(n_route, n_start, demands, Q): continue
            n_cost = calculate_cost(n_route, n_start, D)
            # caso a solução vizinha seja melhor que a solução atual, aceite caso não seja um movimento proibido
            # caso a solução vizinha seja A melhor solução encontrada, aceite
            if (n_cost < next_cost and tabu_list[m] == 0) or n_cost < best_cost:
                next_sol = (n_route, n_start)
                next_cost = n_cost
                movement = m
                if n_cost < best_cost:
                    k = 0
                    best_sol = (n_route, n_start)
                    best_cost = n_cost
        route, start = next_sol

        # atualiza a lista tabu, reduzindo em 1 o número de iterações para cada
        # movimento que esteja na lista
        for i in range(tabu_list.shape[0]):
            for j in range(tabu_list.shape[1]):
                if tabu_list[i,j] > 0:
                    tabu_list[i,j] -= 1
        
        # adiciona na lista tabu o movimento realizado
        tabu_list[movement] = T
    
    return best_sol

@njit
def do_tabu_search(route, start, D, demands, Q):
    '''Seguindo a proposta de Oliveira et al. (2020), executa a BT três vezes
    em sequência, cada vez com um valor diferente para a lista tabu para o critério
    de parada.
    '''
    n = D.shape[0]
    route, start = local_search(route, start, D, demands, Q)

    # movimentos na lista tabu ficarão por n/3 iterações
    # para após 4n iterações sem melhora
    route, start = tabu_search(route, start, D, demands, Q, n // 3, 4 * n)

    # movimentos na lista tabu ficarão por n/6 iterações
    # para após 2n iterações sem melhora
    route, start = tabu_search(route, start, D, demands, Q, n // 6, 2 * n)

    # movimentos na lista tabu ficarão por n²/100 iterações
    # para após n iterações sem melhora
    route, start = tabu_search(route, start, D, demands, Q, n**2 // 100, n)

    return route, start

# hybrid

@njit
def grasp_tabu(route, start, D, demands, Q, alpha=0.3, non_improving_iter=1000):
    '''Aplica uma heurística baseada no GRASP utilizando a busca tabu como
    forma de explorar a vizinhança.
    '''
    # movimentos na lista tabu ficarão por n/3 iterações
    # para após 4n iterações sem melhora
    route, start = tabu_search(route, start, D, demands, Q, D.shape[0] // 3, 4 * D.shape[0])
    best_cost = calculate_cost(route, start, D)
    best_sol = (route, start)

    current_nii = 0
    while current_nii < non_improving_iter:
        route, start = greedy(D, demands, Q, alpha)
        # movimentos na lista tabu ficarão por n/3 iterações
        # para após 4n iterações sem melhora
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
    '''Aplica uma heurística baseada no ILS utilizando a busca tabu como forma
    de explorar a vizinhança.
    '''
    # movimentos na lista tabu ficarão por n/3 iterações
    # para após 4n iterações sem melhora
    route, start = tabu_search(route, start, D, demands, Q, D.shape[0] // 3, 4 * D.shape[0])

    best_sol = (route, start)
    best_cost = calculate_cost(route, start, D)

    current_nii = 0
    while current_nii < non_improving_iter:
        route, start = shake(route, start, D, demands, Q, k, alpha)
        # movimentos na lista tabu ficarão por n/3 iterações
        # para após 4n iterações sem melhora
        route, start = tabu_search(route, start, D, demands, Q, D.shape[0] // 3, 4 * D.shape[0])
        cost = calculate_cost(route, start, D)
        if cost < best_cost:
            current_nii = 0
            best_cost = cost
            best_sol = (route, start)
        else:
            current_nii += 1
    
    return best_sol