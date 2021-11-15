import numpy as np
from numba import njit, types
from greedy import get_next_node
from utils import is_valid

@njit(types.Tuple((types.int32[::1], types.boolean[::1]))(types.int32[::1], types.boolean[::1], types.int32[:, ::1], types.int32[::1], types.int64, types.int64, types.float64))
def shake(route, start, D, demands, Q, k=10, alpha=0):
    '''Altera a solução atual, removendo k vértices e reconstruindo a solução
    de forma semi-gulosa.

    Remove k vértices aleatórios da solução e reconstroi a solução, preenchando
    os gaps criados de utilizando a heurística semi-gulosa baseada no vizinho
    mais próximo.
    '''
    # seleciona k vértices aleatórios para remoção
    remove_idx = np.sort(np.random.choice(np.arange(1, route.shape[0]), k, replace=False))
    visited = np.full_like(demands, True, dtype=np.bool8)
    
    new_route = route.copy()
    new_start = np.full_like(start, False)
    
    # marca os vértices removidos como não visitados
    visited[route[remove_idx]] = False
    
    capacity = 0
    prev_idx = 0

    # marca os gaps onde devem ser inseridos novos vértices
    new_route[remove_idx] = -1
    new_start[1] = True

    # vamos percorrer cada gap criado
    for idx in remove_idx:
        # precisamos percorrer todos os vértices antes do gap
        # as demandas de cada rota são diferentes.
        for i in range(prev_idx+1, idx):
            if capacity + demands[new_route[i]] > Q: # capacidade excedida, vamos iniciar uma nova rota
                new_start[i] = True
            if new_start[i]:
                capacity = 0
            capacity += demands[new_route[i]]
        
        # seleciona um dos vértices removidos que ainda não foi reinserido na solução
        node = get_next_node(D, demands, Q, visited, new_route[idx-1], capacity, alpha)

        # se não foi possível encontrar um nó para inserir na rota atual
        # significa que o veículo da rota atual está cheio, iniciamos uma nova rota
        if node == -1:
            new_start[idx] = True
            capacity = 0
            # agora com o veículo vazio, seleciona um dos vértices
            node = get_next_node(D, demands, Q, visited, new_route[idx-1], capacity, alpha)

        visited[node] = True
        
        new_route[idx] = node
        
        # caso tenha sido iniciada uma nova rota neste índice zeramos o contador da capacidade
        if new_start[idx]:
            capacity = 0

        capacity += demands[node]

        prev_idx = idx
    
    # precisamos agora percorrer todos os vértices após o último gap
    # marcando os novos inícios de rotas para manter a validade da solução.
    for i in range(remove_idx[-1]+1, route.shape[0]):
        if capacity + demands[new_route[i]] > Q:
            new_start[i] = True
        if new_start[i]:
            capacity = 0
        capacity += demands[new_route[i]]

    return new_route, new_start


@njit((types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def rand_two_opt(route, start, demands, Q):
    '''Remove duas arestas e reconecta a rota de forma alternativa.
    
    Gera uma solução vizinha onde foi aplicado o 2-opt.
    '''
    # podemos aplicar este operador apenas intra-rota
    route_starts = np.argwhere(start).flatten()
    p = np.random.randint(0, route_starts.shape[0])
    rs = route_starts[p]
    re = route_starts[p+1] if p+1 < route_starts.shape[0] else route.shape[0]
    
    new_route = route.copy()
    # deve ser selecionada uma rota de tamanho mínimo 4
    # caso contrário não é possível aplicar o operador
    # precisamos de 2 arestas, ou seja, 4 vértices na mesma rota
    if re - rs >= 4:
        i = np.random.randint(rs+1, re-1) # índice aleatório entre o início e o fim da rota
        j = np.random.randint(i+1, re) # índice aleatório entre o índice i e o fim da rota

        # inverte a rota entre os índices i e j
        new_route[i:j] = np.flip(new_route[i:j])

    return new_route, start


@njit((types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def rand_swap(route, start, demands, Q):
    '''Troca dois nós de posição.

    Gera uma solução vizinha onde dois nós estejam trocados de posição.
    '''
    # como uma solução gerada pelo swap pode ser inválida, preciso de alternativas
    # cria e embaralha uma lista de vértices para tentar trocar
    irange = np.arange(1, route.shape[0])
    np.random.shuffle(irange)
    for i in irange:
        # cria e embaralha uma lista de vértices para tentar trocar
        jrange = np.arange(i, route.shape[0])
        np.random.shuffle(jrange)
        for j in jrange:
            if i != j:
                new_route = route.copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]
                # caso a troca gere uma solução válida, retorna essa solução
                if is_valid(new_route, start, demands, Q):
                    return new_route, start
    
    # se não foi possível aplicar o swap vamos retornar a solução sem modificações
    return route, start


@njit((types.int32[::1], types.boolean[::1], types.int32[::1], types.int64))
def perturb(route, start, demands, Q):
    '''Gera uma solução vizinha após a aplicação dos operadores swap e 2-opt aleatorizados.
    '''
    route, start = rand_swap(route, start, demands, Q)
    route, start = rand_two_opt(route, start, demands, Q)
    
    return route, start