import networkx as nx
import numpy 
import threading
import queue

class MyThreading(threading.Thread):
    def __init__(self, outQ, run_AI, comm, k, index, dataset):
        super(MyThreading, self).__init__()
        self.outQ = outQ
        self.run_AI = run_AI
        self.comm = comm
        self.k = k
        self.index = index
        self.dataset = dataset

    # 线程要运行的代码
    def run(self):
        seeds = self.run_AI(self.comm, self.k, self.dataset)
        self.outQ.put(seeds)

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

# 在每个社区内使用AI选点,target
def run_AI(comm, k, dataset):
    comm_G = nx.Graph()
    comm_G.add_nodes_from([int(x) for x in comm])
    f = open('graph/' + dataset + '_graph.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        n1 = int(line.strip().split()[0])
        n2 = int(line.strip().split()[1])
        if (n1 in comm) and (n2 in comm):
            comm_G.add_edge(n1, n2, weight=0.1)

    node_num = comm_G.number_of_nodes()
    comm_I = numpy.ones((node_num, 1))
    comm_A = nx.to_numpy_matrix(comm_G)
    res_I = comm_I

    for i in range(10):
        comm_B = numpy.power(comm_A, i+1)
        comm_C = numpy.matmul(comm_B, comm_I)
        res_I += comm_C

    value = {}

    for j in range(node_num):
        value[comm[j]] = res_I[j, 0]

    # 从大到小排序
    value_sorted = sorted(value.items(), key=lambda item: item[1], reverse=True)
    seeds = []
    for r in range(k):
        seeds.append(value_sorted[r][0])

    return seeds

def mpCommAI(data, n_clusters, dataset, nodeNum):
    communities, select_num = load_data(data, n_clusters, nodeNum)
    worker = []

    for i in range(n_clusters):
        worker.append(MyThreading(queue.Queue(), run_AI, communities[i], select_num[i], i, dataset))
        worker[i].start()

    seed_list = []
    for w in worker:
        seeds = w.outQ.get()
        seed_list = seed_list + seeds
    
    return seed_list

if __name__ == '__main__':
    str1 = "0 2 2 2 0 1 2 0 0 2 0 1 1 1 1 1 2 1 0 2 1 1 0 0 2 0 0 0 1 3 3 0 3 0 0 2 2 2 2 1 0 0 3 0 3 1 1 1 1 3 0 1 2 0 2 2 2 3 2 0 1 2 3 2 2 3 2 0 3 1 0 2 1 1 1 0 2 3 1 1 0 0 3 0 0 2 0 2 1 1 2 1 2 1 0 1 1 2 1 2 2 1 0 2 1 0 0 2 0 0 1 1 0 0 1 2 2 0 3 3 0 1 1 3 0 0 1 2 1 2 2 2 2 2 1 0 1 1 0 0 2 1 0 3 0 3 0 2 3 0 2 0 3 2 0 0 1 0 1 0 2 1 2 1 3 1 3 0 1 2 3 2 2 1 3 0 3 3 2 3 2 1 0 0 1 1 0 0 1 2 0 1 2 0 1 1 2 0 2 2 1 1 0 2 3 1 1 0 0 1 1 2 2 1 3 1 1 2 3 1 2 0 1 2 2 2 2 3 1 0 3 2 1 2 0 2 2 0 1 3 2 0 0 1 3 2 1 2 3 1 1 3 1 3 0 3 2 1 1 1 1 0 2 3 1 1 0 3 1 2 3 1 0 1 2 2 0 1 2 0 2 3 2 2 0 1 1 2 1 0 0 2 3 1 0 2 3 2 0 2 1 2 1 1 1 1 1 0 0 3 2 3 0 1 3 1 2 2 1 2 2 2 1 1 1 2 3 0 2 0 1 2 0 1 2 1 1 1 0 2 0 2 1 2 0 3 0 3 0 2 2 2 1 2 0 0 1 2 0 0 3 1 3 0 1 1 0 0 1 1 2 1 1 1 3 1 0 1 1 1 0 0 1 0 1 3 3 0 2 2 3 1 1 0 0 1 1 3 1 0 0 2 0 3 2 1 2 0 2 0 2 0 0 0 2 2 3 1 2 2 2 2 1 0 0 2 2 1 2 1 2 0 0 2 1 2 3 1 0 1 2 2 2 3 0 1 0 1 1 0 2 1 1 1 1 2 2 0 1 3 1 2 1 0 2 2 3 2 0 2 3 0 0 1 0 1 1 1 1 1 1 1 0 0 2 1 0 2 3 3 1 0 1 1 2 1 1 0 2 2 2 0 1 0 3 0 0 1 1 2 1 1 1 3 2 0 1 0 2 1 0 3 1 2 1 0 1 0 1 3 0 0 2 3 1 0 0 0 0 3 3 3 1 1 0 3 1 1 0 0 3 0 2 0 2 3 0 0 0 1 1 2 2 1 2 1 2 2 2 2 2 0 2 1 2 1 2 1 0 0 0 0 2 1 3 1 0 1 2 2 3 3 0 0 0 1 2 3 1 2 0 0 2 1 2 0 0 2 1 1 2 2 2 0 1 1 0 3 0 2 1 0 0 0 1 3 3 3 2 1 2 2 3 1 3 0 3 1 0 1 0 1 0 3 2 1 2 2 1 2 2 0 1 2 2 1 1 1 2 2 3 2 0 0 0 0 1 0 1 0 0 2 1 1 2 1 1 2 0 0 2 1 0 0 2 2 0 1 1 1 1 2 1 3 1 2 2 0 0 0 2 2 1 0 2 0 1 0 2 1 3 1 3 2 1 2 1 1 0 1 0 1 0 1 0 1 1 0 0 2 2 0 2 0 2 0 1 1 3 0 1 2 1 2 0 2 1 0 2 0 1 3 1 3 2 0 1 2 1 2 0 3 2 0 3 1 3 0 2 0 1 0 1 0 0 3 1 1 3 0 3 2 0 3 1 2 1 2 0 1 3 0 0 0 0 1 1 3 0 2 1 2 3 0 3 1 2 1 1 1 2 3 0 3 2 2 2 1 2 2 1 1 0 3 1 2 0 1 0 2 1 1 1 2 0 1 2 1 2 3 0 2 1 0 3 1 2 0 2 3 1 2 3 3 1 3 1 0 0 0 2 2 1 1 2 0 2 1 1 0 0 0 3 2 1 1 1 0 2 1 1 3 0 0 2 1 0 0 3 3 1 0 2 0 0 0 3 1 1 1 1 1 2 1 1 2 2 2 0 0 2 2 1 1 0 3 1 1 2 0 0 1 3 0 1 0 2 1 0 0 0 1 1 0 2 1 0 3 1 1 2 2 0 0 0 2 3 0 2 1 1 0 1 1 0 1 2 1 0 0 1 3 1 1 0 3 0 0 1 3 1 2 1 1 1 1 1 1 2 1 2 3 2 2 2 0 1 0 2 0 1 2 3 3 0 0 0 1 0 0 1 2 0 3 0 1 0 2 2 1 2 1 3 2 0 2 3 0 3 2 2 2 2 1 1 0 2 0 1 2 0 0 1 2 1 0 0 1 2 3 1 2 3 1 2 1 3 0 1 2 1 2 2 1 3 2 2 0 0 1 3 1 0 2 2 1 1 1 2 1 0 1 2 3 0 1 2 0 0 0 1 1 1 1 0 3 0 1 2 1 3 0 1 1 1 1 2 0 3 1 2 3 3 2 3 2 3 2 1 2 3 2 1 0 0 1 2 3 2 1 0 2 1 0 2 1 0 3 2 1 2 1 0 2 0 1 2 1 1 2 3 3 0 0 2 0 2 1 1 1 1 2 1 1 0 2 1 2 0 1 3 2 1 2 2 0 3 0 0 0 0 2 0 3 2 3 2 1 2 0 0 2 2 1 3 0 0 0 0 2 0 2 1 3 2 2 1 0 2 2 0 2 0 2 1 2 3 1 2 2 1 2 3 0 2 1 2 1 0 3 3 1 1 3 0 3 2 2 1 2 0 1 1 2 0 0 1 1 0 1 2 0 0 0 2 0 0 2 0 2 2 1 1 3 2 1 2 3 0 1 1 1 2 1 3 1 3 1 1 0 3 0 2 1 3 3 0 1 0 3 3 0 3 1 2 2 2 0 2 0 0 1 0 2 2 3 0 2 2 1 3 0 3 1 0 3 1 2 1 1 3 2 2 1 1 2 2 0 3 2 0 0 0 2 1 2 0 1 3 1 1 2 0 3 0 1 2 1 2 1 3 0 3 0 3 2 0 3 3 3 0 1 2 0 3 2 0 0 2 3 1 1 1 0 2 1 0 1 2 2 1 1 0 0 0 3 1 2 1 2 3 0 3 1 0 1 2 3 2 1 2 0 0 0 1 2 0 0 1 2 2 1 1 2 0 1 1 2 3 0 1 2 2 2 3 2 0 3 1 0 3 2 0 1 0 1 1 1 0 3 0 1 1 3 3 1 2 0 3 2 1 0 2 3 3 1 0 2 1 2 1 2 0 1 1 2 0 2 1 3 0 0 1 1 2 0 3 3 0 3 2 1 0 2 1 2 0 1 0 3 2 1 0 2 2 0 0 1 2 1 1 1 2 1 2 3 0 2 1 1 2 2 2 2 1 2 2 2 3 1 2 2 2 1 0 1 0 0 0 2 2 2 1 2 1 2 2 2 2 1 0 2 3 1 1 2 2 2 0 0 3 1 0 2 1 1 1 3 2 2 0 1 1 0 0 1 3 2 3 3 3 1 0 0 0 0 0 1 0 1 1 3 1 1 2 1 3 3 2 1 0 0 1 2 0 1 1 3 0 2 1 0 2 1 2 2 2 3 1 3 1 2 0 0 1 1 2 3 2 3 2 2 3 1 2 0 1 2 2 1 2 1 1 2 1 2 2 2 0 0 0 0 0 1 1 2 1 1 3 1 2 0 1 1 1 1 1 3 2 2 3 1 2 1 2 1 2 1 3 2 1 0 1 2 0 1 2 1 1 3 1 2 3 1 2 2 2 1 1 1 3 1 1 3 0 1 3 2 0 1 0 3 3 2 1 0 1 2 1 3 2 2"
    data = str1.strip().split()
    data = [int(x) for x in data]
    n_clusters = 4
    seeds_list = mpCommAI(data, n_clusters)
    print('seeds_list: ')
    print(seeds_list)
    print('\n')