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
            for alpha in [0, 0.1, 0.2, 0.3]:
                print((i, alpha), end=' ')
                s0 = greedy(D, demands, Q, alpha=alpha)
                if grasp:
                    t0 = time()
                    route, start = metaheuristics.grasp(s0[0], s0[1], D, demands, Q, alpha=0.3, non_improving_iter=4 * D.shape[0])
                    t1 = time()
                    metadata.append({
                        'instance': filepath,
                        'n': D.shape[0],
                        'alpha': alpha,
                        'greedy_cost': calculate_cost(s0[0], s0[1], D),
                        'metaheuristic': 'GRASP',
                        'iteration': i,
                        'time': t1-t0,
                        'cost': calculate_cost(route, start, D),
                        'route': route,
                        'start': start,
                        'valid': is_valid(route, start, demands, Q)
                    })

                if ils:
                    t0 = time()
                    route, start = metaheuristics.ils(s0[0], s0[1], D, demands, Q, k=5, non_improving_iter=4 * D.shape[0])
                    t1 = time()
                    metadata.append({
                        'instance': filepath,
                        'n': D.shape[0],
                        'alpha': alpha,
                        'greedy_cost': calculate_cost(s0[0], s0[1], D),
                        'metaheuristic': 'ILS',
                        'iteration': i,
                        'time': t1-t0,
                        'cost': calculate_cost(route, start, D),
                        'route': route,
                        'start': start,
                        'valid': is_valid(route, start, demands, Q)
                    })

                if simulated_annealing:
                    t0 = time()
                    route, start = metaheuristics.simulated_annealing(s0[0], s0[1], D, demands, Q, T_max=2000, T_min=0.1, alpha=0.95)
                    t1 = time()
                    metadata.append({
                        'instance': filepath,
                        'n': D.shape[0],
                        'alpha': alpha,
                        'greedy_cost': calculate_cost(s0[0], s0[1], D),
                        'metaheuristic': 'Simulated Annealing',
                        'iteration': i,
                        'time': t1-t0,
                        'cost': calculate_cost(route, start, D),
                        'route': route,
                        'start': start,
                        'valid': is_valid(route, start, demands, Q)
                    })
                
                if tabu_search:
                    t0 = time()
                    route, start = metaheuristics.do_tabu_search(s0[0], s0[1], D, demands, Q)
                    t1 = time()
                    metadata.append({
                        'instance': filepath,
                        'n': D.shape[0],
                        'alpha': alpha,
                        'greedy_cost': calculate_cost(s0[0], s0[1], D),
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
                    route, start = metaheuristics.grasp_tabu(s0[0], s0[1], D, demands, Q, alpha=0.3, non_improving_iter=D.shape[0])
                    t1 = time()
                    metadata.append({
                        'instance': filepath,
                        'n': D.shape[0],
                        'alpha': alpha,
                        'greedy_cost': calculate_cost(s0[0], s0[1], D),
                        'metaheuristic': 'GRASP Tabu',
                        'iteration': i,
                        'time': t1-t0,
                        'cost': calculate_cost(route, start, D),
                        'route': route,
                        'start': start,
                        'valid': is_valid(route, start, demands, Q)
                    })

                if ils_tabu:
                    t0 = time()
                    route, start = metaheuristics.ils_tabu(s0[0], s0[1], D, demands, Q, k=5, non_improving_iter=D.shape[0])
                    t1 = time()
                    metadata.append({
                        'instance': filepath,
                        'n': D.shape[0],
                        'alpha': alpha,
                        'greedy_cost': calculate_cost(s0[0], s0[1], D),
                        'metaheuristic': 'ILS Tabu',
                        'iteration': i,
                        'time': t1-t0,
                        'cost': calculate_cost(route, start, D),
                        'route': route,
                        'start': start,
                        'valid': is_valid(route, start, demands, Q)
                    })
            print()
        

    return metadata

if __name__ == '__main__':
    # files = glob.glob('./A-VRP/*.vrp')
    files = ['./A-VRP/A-n32-k5.vrp']
    meta = run_mh(files, iters=10)
    df = pd.DataFrame(meta)
    df.to_csv('results.tsv', sep='\t')
