from cvrp_input import prepare_input
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
    metaheuristics.grasp(D, demands, Q, alpha=0, non_improving_iter=0)
    metaheuristics.ils(D, demands, Q, non_improving_iter=0)
    metaheuristics.simulated_annealing(D, demands, Q, T_max=0.01, alpha=0.99)

    for filepath in files:
        print(filepath)
        D, demands, Q = prepare_input(filepath)
        t0 = time()
        route, start = metaheuristics.grasp(D, demands, Q, alpha=0.5, non_improving_iter=4000)
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

        t0 = time()
        route, start = metaheuristics.ils(D, demands, Q, non_improving_iter=5000)
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

        t0 = time()
        route, start = metaheuristics.simulated_annealing(D, demands, Q, T_max=10, T_min=1, alpha=0.99)
        route, start = local_search(route, start, D, demands, Q)
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
    files = glob.glob('../A-VRP/*.vrp')
    meta = run_mh(files)
    df = pd.DataFrame(meta)
    df.to_csv('results.csv', sep='\t')
