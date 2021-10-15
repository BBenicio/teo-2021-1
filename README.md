# Problema do Roteamento de Veículos Capacitado com Frota Homogênea

Implementamos algumas metaheurísticas para obtenção de soluções para o Problema
do Roteamento de Veículos Capacitado com Frota Homogênea.

Este problema consiste em otimizar as rotas de veículos de forma que atendam a
todos os clientes sem violar a restrição de capacidade de cada veículo.

Utilizamos para testes as instâncias de
[Augerat et al.](https://neo.lcc.uma.es/vrp/vrp-instances/capacitated-vrp-instances/).

As metaheurísticas implementadas são:
- GRASP
- ILS
- Simulated Annealing

Utilizamos a biblioteca Numba JIT para realizar compilação Just-in-Time e
melhorar a performance do código.

Executando o script `main.py` cada metaheurística irá ser rodada para cada uma
das instâncias de entrada.
