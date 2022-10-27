import networkx as nx
import numpy 
import time
from IC import IC

# 在整个图上使用I+AI+A^2I+...+A^10I算法
# Yan, R., Li, D., Wu, W., Du, D. Z., & Wang, Y. (2019). Minimizing influence of rumors by blockers on social networks: algorithms and analysis.

def celf(g, k, candidate, p=0.1, mc=10000, method='random'):
    start_time = time.time()
    marg_gain = [IC(g, [node], p, mc, method=method) for node in candidate]
    Q = sorted(zip(candidate, marg_gain), key=lambda x: x[1], reverse=True)


    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [len(candidate)], [time.time() - start_time]

    for _ in range(k - 1):

        check, node_lookup = False, 0

        while not check:
            node_lookup += 1
            current = Q[0][0]
            Q[0] = (current, IC(g, S + [current], p, mc, method=method) - spread)
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            check = (Q[0][0] == current)
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        Q = Q[1:]

    return (S, SPREAD, timelapse, LOOKUPS)


def run_AI(k):
    G = nx.Graph()
    nodes = [x for x in range(3025)]
    G.add_nodes_from(nodes)
    f = open('graph/acm_graph.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        n1 = int(line.strip().split()[0])
        n2 = int(line.strip().split()[1])
        G.add_edge(n1, n2, weight=0.1)

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
    for r in range(k):
        candidate.append(value_sorted[r][0])

    seeds, spread, timelapse, lookup = celf(G, k, candidate, p=0.1, mc=10000, method='random')

    spreadSum = IC(G, seeds, p=0.1, mc=10000, method='random') 
    return seeds, spreadSum

if __name__ == '__main__':
    f = open('commResult/AI_acm_result.txt', 'w', encoding='utf-8')
    start = time.time()
    k = 20
    seeds, spreadSum = run_AI(k)
    end = time.time()

    print('seeds_list: ' + str(seeds) + '\n')
    print(spreadSum)
    print('运行时间：', end - start)
    f.write('k=' + str(k) + '\n')
    f.write('seeds_list: ' + str(seeds) + '\n')
    f.write('spreadSum: ' + str(spreadSum) + '\n')
    f.write('Time：', end - start + '\n')
    
    
