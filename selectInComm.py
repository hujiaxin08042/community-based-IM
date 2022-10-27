from multiprocessing import Process, Queue, Lock
import networkx as nx
from celf import celf
import numpy as np

# 获取社区
def load_data(data, n_clusters, nodeNum):
    communities = []
    select_num = []
    for i in range(n_clusters):
        communities.append([])
    # 获得社区
    for i, d in enumerate(data):
        communities[d].append(i)
    
    select_num = [len(communities[i]) for i in range(n_clusters)]

    for i in range(n_clusters):
        # N/node_num * k
        select_num[i] = round(len(communities[i]) / nodeNum * 20)
    
    return communities, select_num

# 在每个社区内使用AI选点，串行
def commAI(data, n_clusters, dataset, nodeNum):
    communities, select_num = load_data(data, n_clusters, nodeNum)
    seeds_list = []
    for index in range(n_clusters):
        comm = communities[index]
        k = select_num[index]
        G = nx.Graph()
        G.add_nodes_from([int(x) for x in comm])
        f = open('graph/' + dataset + '_graph.txt', 'r', encoding='utf-8')
        for line in f.readlines():
            n1 = int(line.strip().split()[0])
            n2 = int(line.strip().split()[1])
            if (n1 in comm) and (n2 in comm):
                G.add_edge(n1, n2, weight=0.1)

        num = G.number_of_nodes()
        I = np.ones((num, 1))
        A = nx.to_numpy_matrix(G)
        sigma = I
        for i in range(10):
            B = np.power(A, i+1)
            C = np.matmul(B, I)
            sigma += C
        value = {}

        for j in range(num):
            value[comm[j]] = sigma[j, 0]

        # 从大到小排序
        value_sorted = sorted(value.items(), key=lambda item: item[1], reverse=True)
        seeds = []
        for r in range(k):
            seeds.append(value_sorted[r][0])

        seeds_list = seeds_list + seeds

    return seeds_list