''' 
    Implementation of degree discount heuristic [1] for Independent Cascade model of influence propagation in graph G
    [1] -- Wei Chen et al. Efficient influence maximization in Social Networks (algorithm 4)

    查找要在独立级联模型中传播的初始节点集（无优先级队列）
    Input: 
        - G: networkx 图对象
        - k: 种子节点数
        - p: 传播概率
    Output:
        - S: 种子集
    Note: the routine runs twice slower than using PQ. Implemented to verify results
'''
import networkx as nx
import time
from IC_query import IC
import argparse
import numpy 
import utils
import os

def degreeDiscountIC(G, k):
    d = dict()
    dd = dict()  # degree discount
    t = dict()  # 选择的邻居数量
    S = []  # 种子集
    for u in G.nodes():
        d[u] = sum([G[u][v]['weight'] for v in G[u]])  # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        # dd saves tuples, max function of a tuple compares the first value in the  tuple, if it the same then compare the second,
        # we want to compare only the second, so x being a tuple with x[1] we select the second value of the tuple
        u, ddv = max(dd.items(), key=lambda x: x[1])
#        u, ddv = max(dd.items(), key=lambda (k,v): v)
        dd.pop(u)
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight']  # increase number of selected neighbors
                dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * 0.1
    return S

def degreeDiscountIC1(G, k):
    d = dict()
    dd = dict()  # degree discount
    t = dict()  # 选择的邻居数量
    S = []  # 种子集
    for u in G.nodes():
        d[u] = G.degree(u)
        # d[u] = sum([G[u][v]['weight'] for v in G[u]])  # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        # dd saves tuples, max function of a tuple compares the first value in the  tuple, if it the same then compare the second,
        # we want to compare only the second, so x being a tuple with x[1] we select the second value of the tuple
        u, ddv = max(dd.items(), key=lambda x: x[1])
#        u, ddv = max(dd.items(), key=lambda (k,v): v)
        dd.pop(u)
        S.append(u)
        
        for v in G[u]:
            if v not in S:
                # t[v] += 1
                t[v] += G[u][v]['weight']  # increase number of selected neighbors
                dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * G[u][v]['weight']
    return S


def single_degree_discount(graph, k):
    degree_count = dict(graph.degree)
    topk = []
    # neighborhood_fn = graph.neighbors # if isinstance(graph, nx.Graph) else graph.predecessors
    for _ in range(k):
        node = max(degree_count.items(), key=lambda x: x[1])[0]
        topk.append(node)
        for neighbor in graph[node]:
            if neighbor in degree_count:
                degree_count[neighbor] -= 1
        degree_count.pop(node)
    return topk

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

    if args.dataset == 'wiki':
        args.n_input = 4973
        args.nodeNum = 2405

    start = time.time()

    # 随机生成维数为属性维数的query
    query = numpy.loadtxt('query/' + args.name + '_query.txt', delimiter=',')
    args.query = query
    G = utils.load_graph_query(args.dataset, args.nodeNum, args.query)

    # seeds = degreeDiscountIC1(G, args.k)
    seeds = single_degree_discount(G, args.k)
    print(seeds)
    end = time.time()
    spreadSum = IC(G, seeds, mc=10000, method='pp_random')
    print(spreadSum)
    
 
    # print('dataset: ' + str(args.dataset))
    # f = open('ddisResult/ddis_' + args.dataset + '_' + str(args.k) + '.txt', 'w', encoding='utf-8')
    # print('k: ' + str(args.k) + '\n')
    # print('seeds_list: ' + str(seeds))
    # print('spreadSum: ' + str(spreadSum))
    # print('Time: ', time)
    # f.write('k: ' + str(args.k) + '\n')
    # f.write('seeds_list: ' + str(seeds) + '\n')
    # f.write('spreadSum: ' + str(spreadSum) + '\n')
    # f.write('Time：' + str(time) + '\n')
    # f.close()