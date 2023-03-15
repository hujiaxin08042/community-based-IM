"""
    在整个图上使用I+AI+A^2I+...+A^10I算法
    Yan, R., Li, D., Wu, W., Du, D. Z., & Wang, Y. (2019). Minimizing influence of rumors by blockers on social networks: algorithms and analysis.
    影响概率由余弦相似度计算得出
"""

import networkx as nx
import argparse
import numpy 
import time
import utils
from IC import IC

def celf(g, k, candidate, mc=10000, method='pp_random'):
    start_time = time.time()
    marg_gain = [IC(g, [node], mc, method=method) for node in candidate]
    Q = sorted(zip(candidate, marg_gain), key=lambda x: x[1], reverse=True)


    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [len(candidate)], [time.time() - start_time]

    for _ in range(k - 1):

        check, node_lookup = False, 0
        while not check:
            node_lookup += 1
            current = Q[0][0]
            Q[0] = (current, IC(g, S + [current], mc, method=method) - spread)
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            check = (Q[0][0] == current)
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        Q = Q[1:]

    return (S, SPREAD, timelapse, LOOKUPS)

def run_AI(G, k):
    nodes = list(G.nodes())
    num = G.number_of_nodes()
    I = numpy.ones((num, 1))
    A = nx.to_numpy_matrix(G)
    sigmaI = I

    for i in range(10):
        B = numpy.power(A, i+1)
        C = numpy.matmul(B, I)
        sigmaI += C

    value = {}

    for j in range(num):
        value[nodes[j]] = sigmaI[j, 0]

    # 从大到小排序
    value_sorted = sorted(value.items(), key=lambda item: item[1], reverse=True)
    candidate = []
    for r in range(k*5):
        candidate.append(value_sorted[r][0])

    seeds, spread, timelapse, lookup = celf(G, k, candidate, mc=10000, method='pp_random')

    return seeds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument('--nodeNum', type=int, default=4057)
    # parser.add_argument('--dataset', type=str, default='acm')
    # parser.add_argument('--nodeNum', type=int, default=3025)
    # parser.add_argument('--dataset', type=str, default='cora')
    # parser.add_argument('--nodeNum', type=int, default=2708)
    # parser.add_argument('--dataset', type=str, default='citeseer')
    # parser.add_argument('--nodeNum', type=int, default=3327)
    # parser.add_argument('--dataset', type=str, default='BlogCatalog')
    # parser.add_argument('--nodeNum', type=int, default=10312)
    # parser.add_argument('--dataset', type=str, default='Sinanet')
    # parser.add_argument('--nodeNum', type=int, default=3490)
    # parser.add_argument('--dataset', type=str, default='pubmed')
    # parser.add_argument('--nodeNum', type=int, default=19717)
    parser.add_argument('--k', type=int, default=20)
    args = parser.parse_args()

    start = time.time()
    G = utils.load_graph_pp(args.dataset, args.nodeNum)
    seeds = run_AI(G, args.k)
    end = time.time()

    print('dataset: ' + str(args.dataset))
    f = open('AIResult1/AI_1_' + args.dataset + '_' + str(args.k) + '.txt', 'w', encoding='utf-8')
    spreadSum = IC(G, seeds, mc=10000, method='pp_random') 
    print('k: ' + str(args.k) + '\n')
    print('seeds_list: ' + str(seeds))
    print('spreadSum: ' + str(spreadSum))
    print('Time: ', end - start)
    f.write('k: ' + str(args.k) + '\n')
    f.write('seeds_list: ' + str(seeds) + '\n')
    f.write('spreadSum: ' + str(spreadSum) + '\n')
    f.write('Time：' + str(end - start) + '\n')
    f.close()
    
    