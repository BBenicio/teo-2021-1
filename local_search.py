import numpy as np
from numba import njit, types
from operators import aggregate
from utils import calculate_cost, is_valid

@njit(types.Tuple((types.int32[::1], types.boolean[::1]))(types.int32[::1], types.boolean[::1], types.int32[:, ::1], types.int32[::1], types.int64))
def local_search(route: np.ndarray, start: np.ndarray, D: np.ndarray, demands: np.ndarray, Q: int):
    '''Encontra o ótimo local percorrendo a vizinhança da solução.

    A cada iteração, é selecionado o melhor vizinho até que não seja possível
    obter uma solução melhor.
    '''
    best_cost = calculate_cost(route, start, D)
    best_sol = (route, start)

    cost = best_cost
    
    # enquanto o último vizinho encontrado melhorou a solução
    improved = True
    while improved:
        improved = False
        # calcula a vizinhança
        neighbourhood = aggregate(best_sol[0], best_sol[1]) 
        for r, s in neighbourhood:
            # precisamos verificar a validade da solução, caso seja inválida descarte
            if is_valid(r, s, demands, Q):
                cost = calculate_cost(r, s, D)
                # solução melhor do que a atual, continue buscando
                if cost < best_cost:
                    improved = True
                    best_cost = cost
                    best_sol = (r, s)
    
    return best_sol