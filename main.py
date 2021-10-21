from cvrp_input import prepare_input
from greedy import greedy
from local_search import local_search
from utils import calculate_cost, is_valid
import metaheuristics
from time import time
import glob
import pandas as pd


def run_mh(files: 'list[str]') -> 'list[dict]':
    metadata = []
    D, demands, Q = prepare_input(files[0])
    
    # compila os métodos para não interferir tanto no tempo
    route, start = greedy(D, demands, Q, alpha=0)
    metaheuristics.grasp(D, demands, Q, alpha=0, non_improving_iter=0)
    metaheuristics.ils(route, start, D, demands, Q, non_improving_iter=0)
    metaheuristics.simulated_annealing(route, start, D, demands, Q, T_max=0.01, alpha=0.99)

    for filepath in files:
        print(filepath)
        D, demands, Q = prepare_input(filepath)
        t0 = time()
        route, start = metaheuristics.grasp(D, demands, Q, alpha=0.3, non_improving_iter=4000)
        t1 = time()
        metadata.append({
            'instance': filepath,
            'metaheuristic': 'GRASP',
            'time': t1-t0,
            'cost': calculate_cost(route, start, D),
            'route': route,
            'start': start,
            'valid': is_valid(route, start, demands, Q)
        })

        route, start = greedy(D, demands, Q, alpha=0.3)
        t0 = time()
        route, start = metaheuristics.ils(route, start, D, demands, Q, k=5, non_improving_iter=5000)
        t1 = time()
        metadata.append({
            'instance': filepath,
            'metaheuristic': 'ILS',
            'time': t1-t0,
            'cost': calculate_cost(route, start, D),
            'route': route,
            'start': start,
            'valid': is_valid(route, start, demands, Q)
        })

        route, start = greedy(D, demands, Q, alpha=0.3)
        t0 = time()
        route, start = metaheuristics.simulated_annealing(route, start, D, demands, Q, T_max=10, T_min=1, alpha=0.99)
        t1 = time()
        metadata.append({
            'instance': filepath,
            'metaheuristic': 'Simulated Annealing',
            'time': t1-t0,
            'cost': calculate_cost(route, start, D),
            'route': route,
            'start': start,
            'valid': is_valid(route, start, demands, Q)
        })

    return metadata

if __name__ == '__main__':
    # files = glob.glob('./A-VRP/*.vrp')
    files = ['./A-VRP/A-n32-k5.vrp']
    meta = run_mh(files)
    df = pd.DataFrame(meta)
    df.to_csv('results.csv', sep='\t')
