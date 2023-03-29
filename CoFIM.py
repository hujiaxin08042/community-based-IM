"""
Shang J, Zhou S, Li X, et al. CoFIM: A community-based framework for influence maximization on large-scale networks[J]. Knowledge-Based Systems, 2017, 117: 88-100.
"""

import argparse
import time
import utils
from IC_query import IC
import numpy

class CoFIM():
    def __init__(self, G, dataset, nodeNum):
        self.dataset = dataset
        self.nodeNum = nodeNum
        self.G = G
        self.comm = self.load_comm()

    def load_comm(self):
        communities = utils.louvain(self.G)
        node2comm = dict()  # {节点: 所属社区}
        for index, nodes in enumerate(communities):
            for node in nodes:
                node2comm[int(node)] = index
        return node2comm

    def get_score(self, seed, gamma):
        # 节点邻居
        neigh_node = self.G.neighbors(seed)
        # 节点邻居所属社区
        neigh_comm = set()
        for node in neigh_node:
            neigh_comm.add(self.comm[node])
        return len(list(neigh_node))+len(neigh_comm)*gamma

    def marginal_gain(self, neigh_node, neigh_comm, node, gamma):
        tmp_node = set()
        tmp_comm = set()
        for item in self.G.neighbors(node):
            comm = self.comm[item]
            if item not in neigh_node:
                tmp_node.add(item)
            if comm not in neigh_comm:
                tmp_comm.add(comm)
        return len(tmp_node)+len(tmp_comm)*gamma

    def add_seed(self, seed_set, neigh_node, neigh_comm, node):
        for item in self.G.neighbors(node):
            comm = self.comm[item]
            neigh_node.add(item)
            neigh_comm.add(comm)
        seed_set.append(node)
        return seed_set, neigh_node, neigh_comm

    def seed_selection(self, k, gamma):
        # 计算节点度的平均值
        avg_degree = 2*self.G.number_of_edges()/self.G.number_of_nodes()
        pairs = dict() # {节点: 分数}
        # 选出 10*k 个分数最大的节点
        for node in self.G.nodes():
            if self.G.degree(node) < avg_degree:
                continue
            # 节点度大于平均度时，计算分数
            score = self.get_score(node, gamma)
            # 从小到大排序
            tmp = sorted(pairs.items(), key=lambda item: item[1], reverse=False)
            if len(pairs) >= 10*k and score <= tmp[0][1]:
                continue
            pairs[node] = score
            if len(pairs) > 10*k:
                pairs = dict(sorted(pairs.items(), key=lambda item: item[1], reverse=False)[1:])

        # 从大到小排序
        pairs = dict(sorted(pairs.items(), key=lambda item: item[1], reverse=True))
        updated = [True]*self.nodeNum
        seed_set = []
        neigh_node = set()
        neigh_comm = set()
        for i in range(0, k):
            best_pair = sorted(pairs.items(), key=lambda item: item[1], reverse=True)[0]
            pairs = dict(sorted(pairs.items(), key=lambda item: item[1], reverse=True)[1:])
            # 除了选第一个点
            while(updated[int(best_pair[0])] is not True):
                m_gain = self.marginal_gain(neigh_node, neigh_comm, best_pair[0], gamma)
                updated[best_pair[0]] = True
                pairs[best_pair[0]] = m_gain
                best_pair = sorted(pairs.items(), key=lambda item: item[1], reverse=True)[0]
                pairs = dict(sorted(pairs.items(), key=lambda item: item[1], reverse=True)[1:])
            seed_set, neigh_node, neigh_comm = self.add_seed(seed_set, neigh_node, neigh_comm, best_pair[0])
            updated = [False]*self.nodeNum
        return seed_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--dataset', type=str, default='dblp')
    # parser.add_argument('--dataset', type=str, default='acm')
    # parser.add_argument('--dataset', type=str, default='cora')
    # parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--dataset', type=str, default='BlogCatalog')
    # parser.add_argument('--dataset', type=str, default='Sinanet')
    # parser.add_argument('--dataset', type=str, default='pubmed')
    parser.add_argument('--k', type=int, default=50)
    args = parser.parse_args()

    if args.dataset == 'dblp':
        args.n_input = 334
        args.nodeNum = 4057

    if args.dataset == 'acm':
        args.n_input = 1870
        args.nodeNum = 3025

    if args.dataset == 'cora':
        args.n_input = 1433
        args.nodeNum = 2708

    if args.dataset == 'citeseer':
        args.n_input = 3703
        args.nodeNum = 3327

    if args.dataset == 'BlogCatalog':
        args.n_input = 39
        args.nodeNum = 10312

    if args.dataset == 'Sinanet':
        args.n_input = 10
        args.nodeNum = 3490

    if args.dataset == 'pubmed':
        args.n_input = 500
        args.nodeNum = 19717

    start = time.time()
    # 随机生成维数为属性维数的query
    query = numpy.random.dirichlet(numpy.ones(args.n_input), size=1).reshape(args.n_input,)
    args.query = query
    G = utils.load_graph_query(args.dataset, args.nodeNum, args.query)
    cofim = CoFIM(G, args.dataset, args.nodeNum)
    seeds = cofim.seed_selection(args.k, 3)
    end = time.time()
    
    print('dataset: ' + str(args.dataset))
    f = open('CoFIMResult/CoFIM_' + args.dataset + '_' + str(args.k) + '.txt', 'w', encoding='utf-8')
    spreadSum = IC(G, seeds, mc=10000, method='pp_random') 
    print('k: ' + str(args.k) + '\n')
    print('seeds_list: ' + str(seeds) + '\n')
    print('spreadSum: ' + str(spreadSum) + '\n')
    print('Time: ', end - start)
    f.write('k: ' + str(args.k) + '\n')
    f.write('seeds_list: ' + str(seeds) + '\n')
    f.write('spreadSum: ' + str(spreadSum) + '\n')
    f.write('Time：' + str(end - start) + '\n')
    f.close()