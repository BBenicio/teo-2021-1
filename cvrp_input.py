from io import TextIOWrapper
import numpy as np
from math import sqrt


def read_meta_section(f: TextIOWrapper) -> 'dict[str, str | int]':
    '''Lê os metadados da input.

    Processa as informações do arquivo contidas na seção de metadados.

    Args:
        f (TextIOWrapper): arquivo para ler
    
    Returns:
        dict: dicionário com as informações obtidas (name, comment, type, dimension, edge_weight_type, capacity)
    '''
    line = f.readline()
    metadata = {}
    while 'NODE_COORD_SECTION' not in line:
        split = [ w.strip() for w in line.split(' : ') ]
        key, value = split[0].lower(), split[1]
        metadata[key] = value
        line = f.readline()
    metadata['dimension'] = int(metadata['dimension'])
    metadata['capacity'] = int(metadata['capacity'])
    return metadata


def read_coord_section(f: TextIOWrapper, dimension: int) -> 'list[dict[str, int]]':
    '''Lê as coordenadas dos nós.

    Processa as informações contidas na seção de coordenadas dos nós.

    Args:
        f (TextIOWrapper): arquivo para ler
        dimension: número de nós da instância
    
    Returns:
        list[dict]: posições (x, y) obtidas e demandas (demand) inicializadas
        com zero para cada nó.
    '''
    line = f.readline()
    nodes = [{ 'x': None, 'y': None, 'demand': None } for i in range(dimension)]
    while 'DEMAND_SECTION' not in line:
        node, x, y = [ int(w.strip()) for w in line.strip().split(' ') ]
        nodes[node - 1]['x'] = x
        nodes[node - 1]['y'] = y
        line = f.readline()
    
    return nodes


def read_demand_section(f: TextIOWrapper, nodes: 'list[dict[str, int]]') -> 'list[dict[str, int]]':
    '''Lê as demandas dos clientes.

    Processa as informações na seção de demanda do arquivo.

    Args:
        f (TextIOWrapper): arquivo para ler
        nodes (list[dict]): lista dos nós onde vão ser preenchidas as demandas
    
    Returns:
        list[dict]: lista dos nós com as demandas (demand) obtidas e posições
        (x, y) mantidas.
    '''
    line = f.readline()
    while 'DEPOT_SECTION' not in line:
        node, demand = [ int(w.strip()) for w in line.strip().split(' ') ]
        nodes[node - 1]['demand'] = demand
        line = f.readline()

    return nodes


def read_depot_section(f: TextIOWrapper) -> 'list[dict[str, int]]':
    '''Lê os índices dos depósitos da instância.

    Processa as informações na seção de depósitos do arquivo.

    Args:
        f (TextIOWrapper): arquivo para ler
    
    Returns:
        list[dict]: lista com informações (x, y, demand) dos nós depósito.
    '''
    line = f.readline()
    depots = []
    while '-1' not in line:
        node = int(line.strip())
        if line[node - 1]:
            depots.append(node)
        line = f.readline()
    
    return depots


def read_cvrp(file_path: str) -> 'tuple[dict[str, str | int], list[dict[str, int]], list[dict[str, int]]]':
    '''Lê e processa um arquivo do tipo .vrp.

    Abre o arquivo especificado e o processa obtendo informações sobre a
    instância.

    Args:
        file_path (str): caminho para o arquivo
    
    Returns:
        tuple[dict, list[dict], list[dict]]: metadados, lista de nós e lista de depósitos
    '''
    meta = None
    nodes = None
    depots = None
    with open(file_path, 'r') as f:
        meta = read_meta_section(f)
        nodes = read_coord_section(f, meta['dimension'])
        nodes = read_demand_section(f, nodes)
        depots = read_depot_section(f)
    
    return meta, nodes, depots


def dist(node_a: 'dict[str, int]', node_b: 'dict[str, int]') -> int:
    '''Calcula a distância euclidiana entre dois nós _a_ e _b_.

    Args:
        node_a (dict[str, int]): posição (x, y) do nó _a_
        node_b (dict[str, int]): posição (x, y) do nó _b_
    
    Returns:
        int: distância euclidiana arredondada para inteiro
    '''
    return round(sqrt((node_a['x'] - node_b['x']) ** 2 + (node_a['y'] - node_b['y']) ** 2))


def get_distance_matrix(meta: 'dict[str, str | int]', nodes: 'dict[str, int]') -> np.ndarray:
    '''Calcula a matriz de distâncias do grafo.

    Args:
        meta (dict[str, str | int]): metadados da instância
        nodes (dict[str, int]): informações dos nós do grafo

    Returns:
        np.ndarray: matriz das distâncias entre cada nó.
    '''
    distance_matrix = np.zeros((meta['dimension'], meta['dimension']), dtype=int)
    for i in range(distance_matrix.shape[0]):
        for j in range(i+1, distance_matrix.shape[1]):
            distance_matrix[i, j] = distance_matrix[j, i] = dist(nodes[i], nodes[j])
    
    return distance_matrix


def prepare_input(filepath: str) -> 'tuple[np.ndarray, np.ndarray, int]':
    '''Lê o arquivo e prepara a instância na representação apropriada.

    Cria a matriz de distâncias e o vetor de demandas do arquivo .vrp
    especificado.

    Args:
        filepath (str): caminho para o arquivo

    Returns:
        tuple[np.ndarray, np.ndarray, int]: matriz de distâncias, vetor de demandas, capacidade dos veículos
    '''
    meta, nodes, _ = read_cvrp(filepath)
    D = get_distance_matrix(meta, nodes)
    demands = np.array([n['demand'] for n in nodes])
    Q = meta['capacity']

    return D, demands, Q
    