import numpy as np
from numba import njit, types

@njit(types.int64(types.int32[:,::1], types.int32[::1], types.int64, types.boolean[::1], types.int64, types.int64, types.float64))
def get_next_node(D: np.ndarray, demands: np.ndarray, Q: int, visited: np.ndarray, current_node: int, current_capacity: int = 0, alpha: float = 0):
    '''Calcula o próximo nó baseado na heurística de vizinho mais próximo.

    Retorna o próximo nó para a rota atual de forma semi-gulosa, criando uma
    lista de candidatos cuja restrição é baseada no parâmetro alpha.
    '''
    # um candidato não pode ter sido visitado ou exceder a capacidade do veículo
    candidate_list = (~visited & (current_capacity + demands < Q)).astype(np.bool8)
    if np.any(candidate_list):
        # qual é a distância do candidato mais próximo?
        c_min = np.min(D[current_node][candidate_list])
        # qual é a distância do candidato mais longe?
        c_max = np.max(D[current_node][candidate_list])
        # restringe a lista de candidatos com base nas distâncias e o parâmetro alpha
        # se alpha = 0, a RCL será composta apenas do melhor candidato (mais próximo);
        # se alpha = 1, a RCL será toda a lista de candidatos
        restricted_candidate_list = candidate_list & (D[current_node] <= c_min + alpha * (c_max - c_min))
        if np.any(restricted_candidate_list):
            # escolhe um candidato aleatório da lista restrita
            rcl = np.argwhere(restricted_candidate_list).flatten()
            return np.random.choice(rcl)
        
    # caso não haja um candidato válido
    return -1

@njit(types.Tuple((types.int32[::1], types.boolean[::1]))(types.int32[:,::1], types.int32[::1], types.int64, types.float64))
def greedy(D: np.ndarray, demands: np.ndarray, Q: int, alpha: float = 0):
    '''Constrói uma solução de forma semi-gulosa.

    Cria uma solução utilizando uma heurística de vizinho mais próximo
    semi-gulosa a partir do parâmetro alpha. A solução é retornada como um
    vetor com todas as rotas concatenadas e um segundo vetor marcando as
    posições de início de rota.
    '''
    visited = np.full_like(demands, False, np.bool8)
    visited[0] = True
    
    # vetor que marca os inícios de rotas
    route_start = np.full_like(demands, False, np.bool8)

    # vetor de todas as rotas concatenadas
    route = np.zeros_like(demands)

    i = 1
    while not np.all(visited):
        # inicia uma nova rota, no vértice 0 (depósito), com o veículo vazio, e
        # marca o início no vetor.
        current_node = 0
        current_capacity = 0
        next_node = 0
        route_start[i] = True
        while next_node != -1:
            # heurística de vizinho mais próximo
            next_node = get_next_node(D, demands, Q, visited, current_node, current_capacity, alpha)
            if next_node != -1: # caso seja possível adicionar a esta rota
                current_capacity += demands[next_node]
                visited[next_node] = True
                route[i] = next_node
                current_node = next_node
                i += 1
            # caso não seja possível, finaliza a rota atual e inicia uma próxima
    return route, route_start
