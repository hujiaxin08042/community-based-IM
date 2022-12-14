import networkx as nx
import numpy as np
import multiprocessing as mp
import torch.nn.functional as F
import torch
import torch.nn as nn
import os

# 获取社区
def load_comm(data, n_clusters, dataset, nodeNum, k, query, ratio):
    features = np.loadtxt(os.path.dirname(os.path.abspath(__file__)) + '/data/' + dataset + '.txt', dtype=float)
    communities = []
    select_num = []
    cosines = []
    commSize = []
    for i in range(n_clusters):
        communities.append([])
        select_num.append([])

    # 获得社区
    for i, d in enumerate(data):
        communities[d].append(i)

    for i in range(n_clusters):
        comm = communities[i]
        if len(comm) > 0:
            data = np.asarray([features[i].tolist() for i in comm])
            # 计算属性的平均值
            feature = np.average(data, axis=0)
            # 计算属性平均值和Query的余弦相似度
            cosine = feature.dot(query) / (np.linalg.norm(feature) * np.linalg.norm(query))
            cosines.append(cosine)
            commSize.append(len(comm) / nodeNum)
        else:
            cosines.append(0)
            commSize.append(0)
    
    distribute = F.softmax(torch.tensor(cosines, dtype=torch.float32), dim=0)
    size = F.softmax(torch.tensor(commSize, dtype=torch.float32), dim=0)
    res = F.softmax(distribute + ratio * size, dim=0)

    for i in range(n_clusters):
        select_num[i] = round(float(res[i]) * k)
        # select_num[i] = round(float(size[i]) * k)
    
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

# 度，LT
def run_degree(comm, k, dataset):
    G = nx.Graph()
    G.add_nodes_from([int(x) for x in comm])
    f = open(os.path.dirname(os.path.abspath(__file__)) + '/graph/' + dataset + '_graph.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        n1 = int(line.strip().split()[0])
        n2 = int(line.strip().split()[1])
        if (n1 in comm) and (n2 in comm):
            G.add_edge(n1, n2, weight=0.1)

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
def mpCommAI(data, n_clusters, dataset, nodeNum, k, query, ratio):
    communities, select_num = load_comm(data, n_clusters, dataset, nodeNum, k, query, ratio)
    worker = []

    mp.set_start_method('forkserver', force=True)
    for i in range(n_clusters):    
        worker.append(MyProcess(mp.Queue(), run_AI, communities[i], select_num[i], i, dataset))
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
    communities, select_num = load_comm(data, 4, 'dblp', nodeNum, 20, query)
    print(communities)
    print(select_num)