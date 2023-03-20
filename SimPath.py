"""
Goyal A, Lu W, Lakshmanan L V S. Simpath: An efficient algorithm for influence maximization under the linear threshold model[C]//2011 IEEE 11th international conference on data mining. IEEE, 2011: 211-220.
"""
import numpy as np
import os
import argparse
import threading
import multiprocessing
# import Queue
from multiprocessing import Queue
import networkx as nx
import heapq
import time
import utils
from LT import LT
import numpy
import math

class CELFQueue:
    # create if not exist
    nodes = None
    q = None
    nodes_gain = None

    def __init__(self):
        self.q = []
        self.nodes_gain = {}

    def put(self, node, marginalgain):
        self.nodes_gain[node] = marginalgain
        heapq.heappush(self.q, (-marginalgain, node))

    def update(self, node, marginalgain):
        self.remove(node)
        self.put(node, marginalgain)

    def remove(self, node):
        self.q.remove((-self.nodes_gain[node], node))
        self.nodes_gain[node] = None
        heapq.heapify(self.q)

    def topn(self, n):
        top = heapq.nsmallest(n, self.q)
        top_ = [t[1] for t in top]
        return top_

    def get_gain(self, node):
        return self.nodes_gain[node]

def forward(Q, D, spd, pp, r, W, U, spdW_u, graph):
    x = Q[-1]
    if U is None:
        U = []
    children = list(graph.successors(x))
    count = 0
    while True:
        for child in range(count, len(children)):
            if (children[child] in W) and (children[child] not in Q) and (children[child] not in D[x]):
                y = children[child]
                break
            count = count + 1

        # no such child:
        if count == len(children):
            return Q, D, spd, pp

        if pp * graph[x][y]['weight'] < r:
            D[x].append(y)
        else:
            Q.append(y)
            pp = pp * graph[x][y]['weight']
            spd = spd + pp
            D[x].append(y)
            x = Q[-1]
            for v in U:
                if v not in Q:
                    spdW_u[v] = spdW_u[v] + pp
            children = list(graph.successors(x))
            count = 0

def backtrack(u, r, W, U, spdW_, graph):
    Q = [u]
    spd = 1
    pp = 1
    D = [[] for i in range(graph.number_of_nodes() + 1)]

    while len(Q) != 0:
        Q, D, spd, pp = forward(Q, D, spd, pp, r, W, U, spdW_, graph)
        u = Q.pop()
        D[u] = []
        if len(Q) != 0:
            v = Q[-1]
            if graph[v][u]['weight'] == 0:
                pp = 0
            else:
                pp = pp / graph[v][u]['weight']
    return spd

def simpath_spread(S, r, U, graph, spdW_=None):
    spread = 0
    # W: V-S
    W = set(graph.nodes) - set(S)
    if U is None or spdW_ is None:
        spdW_ = np.zeros(graph.number_of_nodes() + 1)
        # print 'U None'
    for u in S:
        W.add(u)
        # print spdW_[u]
        spread = spread + backtrack(u, r, W, U, spdW_[u], graph)
        # print spdW_[u]
        W.remove(u)
    return spread

def influence_spread_computation_LT(graph, seeds, r=0.01):
    return simpath_spread(seeds, r, None, graph)

# 获取能覆盖所有边的节点
def get_vertex_cover(graph):
    # dv: 存放图中节点的度
    dv = np.zeros(graph.number_of_nodes())
    # e[i,j] = 0: edge (i+1,j+1),(j+1,i+1) checked
    check_array = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    checked = 0

    for i in range(graph.number_of_nodes()):
        dv[i] = graph.degree[i]

    V = set()
    while checked < graph.number_of_edges():
        s = dv.argmax()
        V.add(s)
        # 确保不再选这个节点
        children = graph.successors(s)
        parents = graph.predecessors(s)
        for child in children:
            if check_array[s][child] == 0:
                check_array[s][child] = 1
                checked = checked + 1
        for parent in parents:
            if check_array[parent][s] == 0:
                check_array[parent][s] = 1
                checked = checked + 1
        dv[s] = -1
    
    return list(V)

def simpath(graph, k, r, l):
    # 覆盖所有边的节点集
    C = set(get_vertex_cover(graph))
    # 图中的所有节点集
    V = set(graph.nodes())

    V_C = V - C
    # spread[x] is spd of S + x
    spread = np.zeros(graph.number_of_nodes())
    spdV_ = np.ones((graph.number_of_nodes(), graph.number_of_nodes()))
    # 计算在C中的节点的spread
    for u in C:
        U =  V_C & set(graph.predecessors(u))
        spread[u] = simpath_spread(set([u]), r, U, graph, spdV_)
    
    # 计算不在C中的节点的spread
    for v in V_C:
        v_children = graph.successors(v)
        for child in v_children:
            spread[v] = spread[v] + spdV_[child][v] * graph[v][child]['weight']
        spread[v] = spread[v] + 1
    
    celf = CELFQueue()
    # 将所有节点放入celf队列
    # spread[v] is the marginal gain at this time
    for node in range(0, graph.number_of_nodes()):
        celf.put(node, spread[node])
    S = set()
    W = V # 所有节点
    spd = 0
    # mark the node that checked before during the same Si
    checked = np.zeros(graph.number_of_nodes())

    while len(S) < k:
        U = celf.topn(l)
        spdW_ = np.ones((graph.number_of_nodes(), graph.number_of_nodes()))
        spdV_x = np.zeros(graph.number_of_nodes())
        simpath_spread(S, r, U, graph, spdW_=spdW_)
        for x in U:
            for s in S:
                spdV_x[x] = spdV_x[x] + spdW_[s][x]
        for x in U:
            if checked[x] != 0:
            # if checked[x] == 0:
                S.add(x)
                W = W - set([x])
                spd = spread[x]
                # checked = np.zeros(graph.number_of_nodes())
                if (-celf.nodes_gain[x], x) in celf.q:
                    celf.remove(x)
                break
            else:
                spread[x] = backtrack(x, r, W, None, None, graph) + spdV_x[x]
                checked[x] = 1
                if (-celf.nodes_gain[x], x) in celf.q:
                    celf.update(x, spread[x] - spd)
    return S

def simpath1(graph, r):
    # 覆盖所有边的节点集
    C = set(get_vertex_cover(graph))
    # 图中的所有节点集
    V = set(graph.nodes())

    V_C = V - C
    # spread[x] is spd of S + x
    spread = np.zeros(graph.number_of_nodes())
    spdV_ = np.ones((graph.number_of_nodes(), graph.number_of_nodes()))
    # 计算在C中的节点的spread
    for u in C:
        U = V_C & set(graph.predecessors(u))
        spread[u] = simpath_spread(set([u]), r, U, graph, spdV_)

    # 计算不在C中的节点的spread
    for v in V_C:
        v_children = graph.successors(v)
        for child in v_children:
            spread[v] = spread[v] + spdV_[child][v] * graph[v][child]['weight']
        spread[v] = spread[v] + 1

    return spread

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='dblp')
    # parser.add_argument('--dataset', type=str, default='acm')
    # parser.add_argument('--dataset', type=str, default='cora')
    # parser.add_argument('--dataset', type=str, default='citeseer')
    # parser.add_argument('--dataset', type=str, default='BlogCatalog')
    # parser.add_argument('--dataset', type=str, default='Sinanet')
    # parser.add_argument('--dataset', type=str, default='pubmed')
    # parser.add_argument('--dataset', type=str, default='wiki')
    parser.add_argument('--k', type=int, default=10)
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
    G = utils.load_DiGraph_query(args.dataset, args.nodeNum, args.n_input)
    seeds = simpath(G, args.k, 0.001, 7)
    print(seeds)
    end = time.time()

    print('dataset: ' + str(args.dataset))
    f = open('simpathResult/simpath_' + args.dataset + '_' + str(args.k) + '.txt', 'a', encoding='utf-8')
    spreadSum = LT(G, seeds, mc=10000, method='pp_random') 
    print('k: ' + str(args.k) + '\n')
    print('seeds_list: ' + str(seeds))
    print('spreadSum: ' + str(spreadSum))
    print('Time: ', end - start)
    f.write('k: ' + str(args.k) + '\n')
    f.write('seeds_list: ' + str(seeds) + '\n')
    f.write('spreadSum: ' + str(spreadSum) + '\n')
    f.write('Time：' + str(end - start) + '\n')
    f.close()
        
