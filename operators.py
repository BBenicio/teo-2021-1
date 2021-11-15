import numpy as np
from numba import njit, types

@njit((types.int32[::1], types.boolean[::1]))
def swap(route: np.ndarray, start: np.ndarray):
    '''Troca dois nós de posição.

    Gera uma lista de soluções vizinhas onde dois nós estão trocados de posição
    na rota.
    '''
    neighbourhood = []
    # percorre todos os pares de nós e os troca, gerando uma solução vizinha
    # possivelmente inválida, ou seja, que viole a restrição de capacidade
    for i in range(1, route.shape[0]):
        for j in range(i+1, route.shape[0]):
            new_route = route.copy()
            new_route[i], new_route[j] = new_route[j], new_route[i]
            
            neighbourhood.append((new_route, start))
    
    return neighbourhood


@njit((types.int32[::1], types.boolean[::1]))
def two_opt(route, start):
    '''Remove duas arestas e reconecta a rota de forma alternativa.

    Gera uma lista de soluções vizinhas onde foi aplicado o operador 2-opt.
    '''
    neighbourhood = []
    # encontra os índices no vetor onde são iniciadas as rotas
    # precisamos percorrer cada rota individualmente
    route_starts = np.argwhere(start).flatten()
    for p in range(route_starts.shape[0]):
        # índice do início da rota
        rs = route_starts[p]
        # índice do final da rota
        re = route_starts[p+1] if p+1 < route_starts.shape[0] else route.shape[0]
        
        # percorre todo par de arestas i, j na rota atual
        for i in range(rs+1, re):
            # a aresta i é a aresta entre o vértice na posição i e o vértice na posição i-1
            for j in range(i+1, re):
                # a aresta j é a aresta entre o vértice na posição j e o vértice na posição j-1
                new_route = route.copy()
                # inverte a rota entre os índices selecionados
                new_route[i:j] = np.flip(new_route[i:j])

                neighbourhood.append((new_route, start))
    
    return neighbourhood


@njit((types.int32[::1], types.boolean[::1]))
def aggregate(route, start):
    '''Agrega o resultado das vizinhanças swap e 2-opt em uma única lista.

    Gera uma lista de soluções vizinhas com as vizinhanças swap e 2-opt.
    '''
    return swap(route, start) + two_opt(route, start)


# TABU variants

@njit((types.int32[::1], types.boolean[::1]))
def tabu_swap(route, start):
    '''Troca dois nós de posição, retornando também os índices do movimento.

    Gera uma lista de soluções vizinhas onde dois nós estão trocados de posição
    na rota. A lista contém os índices i, j que indicam quais vértices foram
    movimentados para gerar cada solução.
    '''
    neighbourhood = []
    for i in range(1, route.shape[0]):
        for j in range(i+1, route.shape[0]):
            new_route = route.copy()
            new_route[i], new_route[j] = new_route[j], new_route[i]
            # retorna os índices i, j para serem adicionados na lista tabu caso esse vizinho seja aceito
            neighbourhood.append((new_route, start, (i, j)))
    
    return neighbourhood


@njit((types.int32[::1], types.boolean[::1]))
def tabu_two_opt(route, start):
    '''Remove duas arestas e reconecta a rota de forma alternativa, retornando
    também os índices do movimento.

    Gera uma lista de soluções vizinhas onde foi aplicado o operador 2-opt.
    A lista contém os índices i, j que indicam quais arestas foram
    movimentadas para gerar cada solução.
    '''
    neighbourhood = []
    route_starts = np.argwhere(start).flatten()
    for p in range(route_starts.shape[0]):
        rs = route_starts[p]
        re = route_starts[p+1] if p+1 < route_starts.shape[0] else route.shape[0]
        
        for i in range(rs+1, re):
            for j in range(i+1, re):
                new_route = route.copy()
                new_route[i:j] = np.flip(new_route[i:j])

                # retorna os índices i, j para serem adicionados na lista tabu caso esse vizinho seja aceito
                neighbourhood.append((new_route, start, (i, j)))
    
    return neighbourhood
