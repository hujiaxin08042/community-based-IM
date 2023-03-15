import networkx as nx
import numpy as np
import multiprocessing as mp
import torch.nn.functional as F
import torch
import torch.nn as nn
import os

# 获取社区
def load_comm(comm_data, n_clusters, dataset, nodeNum, k, comm_select_num):

    communities = [[] for i in range(n_clusters)]
    for i, d in enumerate(comm_data):
        communities[d].append(i)
    res = F.softmax(torch.tensor(comm_select_num, dtype=torch.float32), dim=0)
    print("comm_select_num:", res)
    select_num = [round(float(res[i]) * k) for i in range(n_clusters)]
    return communities, select_num

# AI选点，IC
def run_AI(comm, k, dataset):
    G = nx.Graph()
    G.add_nodes_from([int(x) for x in comm])
    f = open(os.path.dirname(os.path.abspath(__file__)) + '/similarity/' + dataset + '_cosine.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        n1 = int(line.strip().split()[0])
        n2 = int(line.strip().split()[1])
        weight = float(line.strip().split()[2])
        if (n1 in comm) and (n2 in comm):
            G.add_edge(n1, n2, weight=weight)

    commNodeNum = G.number_of_nodes()
    I = np.ones((commNodeNum, 1))
    A = nx.to_numpy_matrix(G)
    res_I = I

    for i in range(10):
        B = np.power(A, i+1)
        C = np.matmul(B, I)
        res_I += C

    value = {}

    for j in range(commNodeNum):
        value[comm[j]] = res_I[j, 0]

    # 从大到小排序
    value_sorted = sorted(value.items(), key=lambda item: item[1], reverse=True)
    seeds = []
    if len(value_sorted) < k:
        for r in range(len(value_sorted)):
            seeds.append(value_sorted[r][0])
    else:        
        for r in range(k):
            seeds.append(value_sorted[r][0])

    return seeds

# 度，自适应
def run_degree(comm, k, dataset):
    seeds = []
    G = nx.Graph()
    G.add_nodes_from([int(x) for x in comm])
    f = open(os.path.dirname(os.path.abspath(__file__)) + '/similarity/' + dataset + '_pp.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        n1 = int(line.strip().split()[0])
        n2 = int(line.strip().split()[1])
        weight = float(line.strip().split()[2])
        if (n1 in comm) and (n2 in comm):
            G.add_edge(n1, n2, weight=weight)

    for i in range(k):
        if G.number_of_nodes() > 0:
            res = list(G.degree())
            res = sorted(res, key=lambda item: item[1], reverse=True)
            node = res[0][0]
            seeds.append(node)
            G.remove_node(node)
        else:
            break

    return seeds

# 覆写进程
class MyProcess(mp.Process):
    def __init__(self, outQ, run_AI, comm, k, index, dataset):
        super().__init__()
        self.outQ = outQ
        self.run_AI = run_AI
        self.comm = comm
        self.k = k
        self.index = index
        self.dataset = dataset

    # 进程要运行的代码
    def run(self):
        seeds = self.run_AI(self.comm, self.k, self.dataset)

        self.outQ.put(seeds)

# 在每个社区内使用AI选点，进程并行
def mpCommAI(data, n_clusters, dataset, nodeNum, k, comm_select_num):
    communities, select_num = load_comm(data, n_clusters, dataset, nodeNum, k, comm_select_num)
    worker = []

    mp.set_start_method('forkserver', force=True)
    for i in range(n_clusters):    
        # worker.append(MyProcess(mp.Queue(), run_AI, communities[i], select_num[i], i, dataset))
        worker.append(MyProcess(mp.Queue(), run_degree, communities[i], select_num[i], i, dataset))
        worker[i].start()
    
    seed_list = []
    for w in worker:
        seeds = w.outQ.get()
        seed_list = seed_list + seeds
            
    for i in worker:
        i.join()
    
    return seed_list

if __name__ == '__main__':
    str1 = "0 2 2 2 0 1 2 0 0 2 0 1 1 1 1 1 2 1 0 2 1 1 0 0 2 0 0 0 1 3 3 0 3 0 0 2 2 2 2 1 0 0 3 0 3 1 1 1 1 3 0 1 2 0 2 2 2 3 2 0 1 2 3 2 2 3 2 0 3 1 0 2 1 1 1 0 2 3 1 1 0 0 3 0 0 2 0 2 1 1 2 1 2 1 0 1 1 2 1 2 2 1 0 2 1 0 0 2 0 0 1 1 0 0 1 2 2 0 3 3 0 1 1 3 0 0 1 2 1 2 2 2 2 2 1 0 1 1 0 0 2 1 0 3 0 3 0 2 3 0 2 0 3 2 0 0 1 0 1 0 2 1 2 1 3 1 3 0 1 2 3 2 2 1 3 0 3 3 2 3 2 1 0 0 1 1 0 0 1 2 0 1 2 0 1 1 2 0 2 2 1 1 0 2 3 1 1 0 0 1 1 2 2 1 3 1 1 2 3 1 2 0 1 2 2 2 2 3 1 0 3 2 1 2 0 2 2 0 1 3 2 0 0 1 3 2 1 2 3 1 1 3 1 3 0 3 2 1 1 1 1 0 2 3 1 1 0 3 1 2 3 1 0 1 2 2 0 1 2 0 2 3 2 2 0 1 1 2 1 0 0 2 3 1 0 2 3 2 0 2 1 2 1 1 1 1 1 0 0 3 2 3 0 1 3 1 2 2 1 2 2 2 1 1 1 2 3 0 2 0 1 2 0 1 2 1 1 1 0 2 0 2 1 2 0 3 0 3 0 2 2 2 1 2 0 0 1 2 0 0 3 1 3 0 1 1 0 0 1 1 2 1 1 1 3 1 0 1 1 1 0 0 1 0 1 3 3 0 2 2 3 1 1 0 0 1 1 3 1 0 0 2 0 3 2 1 2 0 2 0 2 0 0 0 2 2 3 1 2 2 2 2 1 0 0 2 2 1 2 1 2 0 0 2 1 2 3 1 0 1 2 2 2 3 0 1 0 1 1 0 2 1 1 1 1 2 2 0 1 3 1 2 1 0 2 2 3 2 0 2 3 0 0 1 0 1 1 1 1 1 1 1 0 0 2 1 0 2 3 3 1 0 1 1 2 1 1 0 2 2 2 0 1 0 3 0 0 1 1 2 1 1 1 3 2 0 1 0 2 1 0 3 1 2 1 0 1 0 1 3 0 0 2 3 1 0 0 0 0 3 3 3 1 1 0 3 1 1 0 0 3 0 2 0 2 3 0 0 0 1 1 2 2 1 2 1 2 2 2 2 2 0 2 1 2 1 2 1 0 0 0 0 2 1 3 1 0 1 2 2 3 3 0 0 0 1 2 3 1 2 0 0 2 1 2 0 0 2 1 1 2 2 2 0 1 1 0 3 0 2 1 0 0 0 1 3 3 3 2 1 2 2 3 1 3 0 3 1 0 1 0 1 0 3 2 1 2 2 1 2 2 0 1 2 2 1 1 1 2 2 3 2 0 0 0 0 1 0 1 0 0 2 1 1 2 1 1 2 0 0 2 1 0 0 2 2 0 1 1 1 1 2 1 3 1 2 2 0 0 0 2 2 1 0 2 0 1 0 2 1 3 1 3 2 1 2 1 1 0 1 0 1 0 1 0 1 1 0 0 2 2 0 2 0 2 0 1 1 3 0 1 2 1 2 0 2 1 0 2 0 1 3 1 3 2 0 1 2 1 2 0 3 2 0 3 1 3 0 2 0 1 0 1 0 0 3 1 1 3 0 3 2 0 3 1 2 1 2 0 1 3 0 0 0 0 1 1 3 0 2 1 2 3 0 3 1 2 1 1 1 2 3 0 3 2 2 2 1 2 2 1 1 0 3 1 2 0 1 0 2 1 1 1 2 0 1 2 1 2 3 0 2 1 0 3 1 2 0 2 3 1 2 3 3 1 3 1 0 0 0 2 2 1 1 2 0 2 1 1 0 0 0 3 2 1 1 1 0 2 1 1 3 0 0 2 1 0 0 3 3 1 0 2 0 0 0 3 1 1 1 1 1 2 1 1 2 2 2 0 0 2 2 1 1 0 3 1 1 2 0 0 1 3 0 1 0 2 1 0 0 0 1 1 0 2 1 0 3 1 1 2 2 0 0 0 2 3 0 2 1 1 0 1 1 0 1 2 1 0 0 1 3 1 1 0 3 0 0 1 3 1 2 1 1 1 1 1 1 2 1 2 3 2 2 2 0 1 0 2 0 1 2 3 3 0 0 0 1 0 0 1 2 0 3 0 1 0 2 2 1 2 1 3 2 0 2 3 0 3 2 2 2 2 1 1 0 2 0 1 2 0 0 1 2 1 0 0 1 2 3 1 2 3 1 2 1 3 0 1 2 1 2 2 1 3 2 2 0 0 1 3 1 0 2 2 1 1 1 2 1 0 1 2 3 0 1 2 0 0 0 1 1 1 1 0 3 0 1 2 1 3 0 1 1 1 1 2 0 3 1 2 3 3 2 3 2 3 2 1 2 3 2 1 0 0 1 2 3 2 1 0 2 1 0 2 1 0 3 2 1 2 1 0 2 0 1 2 1 1 2 3 3 0 0 2 0 2 1 1 1 1 2 1 1 0 2 1 2 0 1 3 2 1 2 2 0 3 0 0 0 0 2 0 3 2 3 2 1 2 0 0 2 2 1 3 0 0 0 0 2 0 2 1 3 2 2 1 0 2 2 0 2 0 2 1 2 3 1 2 2 1 2 3 0 2 1 2 1 0 3 3 1 1 3 0 3 2 2 1 2 0 1 1 2 0 0 1 1 0 1 2 0 0 0 2 0 0 2 0 2 2 1 1 3 2 1 2 3 0 1 1 1 2 1 3 1 3 1 1 0 3 0 2 1 3 3 0 1 0 3 3 0 3 1 2 2 2 0 2 0 0 1 0 2 2 3 0 2 2 1 3 0 3 1 0 3 1 2 1 1 3 2 2 1 1 2 2 0 3 2 0 0 0 2 1 2 0 1 3 1 1 2 0 3 0 1 2 1 2 1 3 0 3 0 3 2 0 3 3 3 0 1 2 0 3 2 0 0 2 3 1 1 1 0 2 1 0 1 2 2 1 1 0 0 0 3 1 2 1 2 3 0 3 1 0 1 2 3 2 1 2 0 0 0 1 2 0 0 1 2 2 1 1 2 0 1 1 2 3 0 1 2 2 2 3 2 0 3 1 0 3 2 0 1 0 1 1 1 0 3 0 1 1 3 3 1 2 0 3 2 1 0 2 3 3 1 0 2 1 2 1 2 0 1 1 2 0 2 1 3 0 0 1 1 2 0 3 3 0 3 2 1 0 2 1 2 0 1 0 3 2 1 0 2 2 0 0 1 2 1 1 1 2 1 2 3 0 2 1 1 2 2 2 2 1 2 2 2 3 1 2 2 2 1 0 1 0 0 0 2 2 2 1 2 1 2 2 2 2 1 0 2 3 1 1 2 2 2 0 0 3 1 0 2 1 1 1 3 2 2 0 1 1 0 0 1 3 2 3 3 3 1 0 0 0 0 0 1 0 1 1 3 1 1 2 1 3 3 2 1 0 0 1 2 0 1 1 3 0 2 1 0 2 1 2 2 2 3 1 3 1 2 0 0 1 1 2 3 2 3 2 2 3 1 2 0 1 2 2 1 2 1 1 2 1 2 2 2 0 0 0 0 0 1 1 2 1 1 3 1 2 0 1 1 1 1 1 3 2 2 3 1 2 1 2 1 2 1 3 2 1 0 1 2 0 1 2 1 1 3 1 2 3 1 2 2 2 1 1 1 3 1 1 3 0 1 3 2 0 1 0 3 3 2 1 0 1 2 1 3 2 2"
    data = str1.strip().split()
    data = [int(x) for x in data]
    nodeNum = len(data)
    query = np.random.dirichlet(np.ones(334), size=1).reshape(334,)
    communities, select_num = load_comm(data, 4, 'dblp', nodeNum, 20, query, 0.5)
    print(communities)
    print(select_num)