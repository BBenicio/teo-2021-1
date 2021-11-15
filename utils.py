import numpy as np
from numba import njit, types

def print_routes(route, start, D, demands):
    r = 0
    cost = 0
    prev = 0
    capacity = 0
    for i in range(1, route.shape[0]):
        if start[i]:
            if r > 0: print(f'\t({capacity})', end='')
            cost += D[prev, 0] # return to depot
            capacity = 0
            prev = 0
            r += 1
            print(f'\nRoute #{r}', end=': ')

        n = route[i]
        print(n, end=' ')
        capacity += demands[n]
        cost += D[prev, n]
        prev = n

    cost += D[prev, 0] # return to depot
    print(f'\t({capacity})')
    print(f'\nCost: {cost}')


@njit(types.boolean(types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def is_valid(route, start, demands, Q):
    '''Verifica a validade da solução de acordo com as restrições do problema

    A solução é válida se, e somente se:
    1. Todos os nós são visitados
    2. Nenhum nó é visitado mais de uma vez
    3. Nenhuma rota possui clientes cuja soma das demandas exceda a capacidade dos veículos
    '''
    current_capacity = 0
    visited = np.zeros_like(demands)
    visited[0] = 1
    if not start[1]: return False
    for i in range(1, route.shape[0]):
        # contador de visitas por nó
        visited[route[i]] += 1
        
        # nova rota iniciada, verifica a capacidade e zera o contador de capacidade
        if start[i]:
            if current_capacity > Q: # capacidade do veículo excedida, solução inválida
                return False
            current_capacity = 0

        current_capacity += demands[route[i]]
    
    # última rota foi percorrida, devemos verificar se todos os nós foram visitados uma única vez
    # e se a capacidade do veículo foi respeitada na última rota
    return np.all(visited == 1) and current_capacity <= Q


@njit(types.int64(types.int32[::1], types.boolean[::1], types.int32[:, ::1]))
def calculate_cost(route, start, D):
    '''Calcula o custo da solução somando as distâncias que cada veículo percorre.

    Em uma rota v0->v1->v2->v3->v0, devem ser somadas as distâncias entre cada
    par de vértices consecutivos. Assim o custo da rota é dado por:
    C = d(v0,v1) + d(v1,v2) + d(v2,v3) + d(v3,v0)
    '''
    cost = 0 # inicia com custo 0
    prev = 0 # depósito é o vértice anterior
    for i in range(1, route.shape[0]):
        # nova rota iniciada, soma o custo de retornar ao depósito e marca o depósito como sendo o vértice anterior
        if start[i]:
            cost += D[prev, 0]
            prev = 0

        n = route[i] # vértice atual
        cost += D[prev, n] # soma custo de ir do vértice anterior para o atual
        prev = n # atualiza vértice anterior
    
    cost += D[prev, 0] # chegou ao fim da última rota, precisamos voltar ao depósito
    return cost