from cvrp_input import prepare_input
from greedy import greedy
from utils import calculate_cost, is_valid
import metaheuristics
from time import time
import glob
import pandas as pd


def run_mh(files: 'list[str]', grasp=True, ils=True, simulated_annealing=True, tabu_search=True, grasp_tabu=True, ils_tabu=True, iters=1) -> 'list[dict]':
    metadata = []
    D, demands, Q = prepare_input(files[0])
    
    for filepath in files:
        print(filepath)
        D, demands, Q = prepare_input(filepath)
        for i in range(iters):
            if grasp:
                t0 = time()
                route, start = metaheuristics.grasp(D, demands, Q, alpha=0.3, non_improving_iter=4 * D.shape[0])
                t1 = time()
                metadata.append({
                    'instance': filepath,
                    'metaheuristic': 'GRASP',
                    'iteration': i,
                    'time': t1-t0,
                    'cost': calculate_cost(route, start, D),
                    'route': route,
                    'start': start,
                    'valid': is_valid(route, start, demands, Q)
                })

            if ils:
                route, start = greedy(D, demands, Q, alpha=0.3)
                t0 = time()
                route, start = metaheuristics.ils(route, start, D, demands, Q, k=5, non_improving_iter=4 * D.shape[0])
                t1 = time()
                metadata.append({
                    'instance': filepath,
                    'metaheuristic': 'ILS',
                    'iteration': i,
                    'time': t1-t0,
                    'cost': calculate_cost(route, start, D),
                    'route': route,
                    'start': start,
                    'valid': is_valid(route, start, demands, Q)
                })

            if simulated_annealing:
                route, start = greedy(D, demands, Q, alpha=0.3)
                t0 = time()
                route, start = metaheuristics.simulated_annealing(route, start, D, demands, Q, T_max=2000, T_min=0.1, alpha=0.95)
                t1 = time()
                metadata.append({
                    'instance': filepath,
                    'metaheuristic': 'Simulated Annealing',
                    'iteration': i,
                    'time': t1-t0,
                    'cost': calculate_cost(route, start, D),
                    'route': route,
                    'start': start,
                    'valid': is_valid(route, start, demands, Q)
                })
            
            if tabu_search:
                route, start = greedy(D, demands, Q, alpha=0.3)
                t0 = time()
                route, start = metaheuristics.do_tabu_search(route, start, D, demands, Q)
                t1 = time()
                metadata.append({
                    'instance': filepath,
                    'metaheuristic': 'Tabu Search',
                    'iteration': i,
                    'time': t1-t0,
                    'cost': calculate_cost(route, start, D),
                    'route': route,
                    'start': start,
                    'valid': is_valid(route, start, demands, Q)
                })
            
            if grasp_tabu:
                t0 = time()
                route, start = metaheuristics.grasp_tabu(D, demands, Q, alpha=0.3, non_improving_iter=D.shape[0])
                t1 = time()
                metadata.append({
                    'instance': filepath,
                    'metaheuristic': 'GRASP Tabu',
                    'iteration': i,
                    'time': t1-t0,
                    'cost': calculate_cost(route, start, D),
                    'route': route,
                    'start': start,
                    'valid': is_valid(route, start, demands, Q)
                })

            if ils_tabu:
                route, start = greedy(D, demands, Q, alpha=0.3)
                t0 = time()
                route, start = metaheuristics.ils_tabu(route, start, D, demands, Q, k=5, non_improving_iter=D.shape[0])
                t1 = time()
                metadata.append({
                    'instance': filepath,
                    'metaheuristic': 'ILS Tabu',
                    'iteration': i,
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
