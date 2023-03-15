"""
[1] Rahimkhani K, Aleahmad A, Rahgozar M, et al. A fast algorithm for finding most influential people based on the linear threshold model[J]. Expert Systems with Applications, 2015, 42(3): 1353-1361.
"""
import networkx as nx
import math
import argparse
import time
import utils
import os
from LT import LT

# 创建社区图，每个节点表示一个社区
def createCommNet(graph, communities):
    G = nx.DiGraph()
    node2comm = {}
    count = 0
    for i, data in enumerate(communities):
        count += 1
        for node in data:
            if node in node2comm:
                v = node2comm.get(node)
                v.append(i)
                node2comm[node] = v
            else:
                node2comm[node] = [i]

    G.add_nodes_from([i for i in range(count)])

    for edge in graph.edges():
        node1 = edge[0]
        node2 = edge[1]
        comm1_list = node2comm.get(node1)
        comm2_list = node2comm.get(node2)
        if comm1_list and comm2_list and comm1_list != comm2_list:
            for comm1 in comm1_list:
                for comm2 in comm2_list:
                    G.add_edge(comm1, comm2)            

    return G

# 计算网络中的节点的介数中心性，并降序输出
def topNBetweeness(G):
    score = nx.betweenness_centrality(G)
    score = sorted(score.items(), key=lambda item: item[1], reverse=True)
    comm_btn = [node[0] for node in score]
    return comm_btn
    # f = open("data/NetHEPT_BC.txt", 'w', encoding='utf-8')
    # for node in score:
    #     f.write(str(node[0]) + ' ')

# 计算应从每个社区中选取多少点
def calNumOfNode(alpha, beta):
    f1 = open('data/NetHEPT_slpa.txt', 'r', encoding='utf-8')
    f2 = open('data/NetHEPT_BC.txt', 'r', encoding='utf-8')
    f3 = open('data/NetHEPT_selectedComm.txt', 'w', encoding='utf-8')
    for line in f2.readlines():
        communities = line.strip().split()[0: 255]
    communities = [int(x) for x in communities]

    print(len(communities))
    maxV = 0
    minV = 0
    result = dict()
    for i, line in enumerate(f1):
        if i in communities:
            f3.write(str(i) + ' ' + line)
            num = len(line.strip().split())
            result[i] = num
            if i == 0:
                maxV = num
                minV = num
            maxV = max(maxV, num)
            minV = min(minV, num)

    for k, v in result.items():
        result[k] = v / (maxV - minV) * beta + alpha
    print(result)
    f3.close()
    return result

# 根据度在Top255个社区出选出候选节点
def getSelectedNodes(nodeNum):
    f1 = open('../data/NetHEPT_graph.txt', 'r', encoding='utf-8')
    f2 = open('data/NetHEPT_selectedComm.txt', 'r', encoding='utf-8')
    f3 = open('data/NetHEPT_selectedNode.txt', 'w', encoding='utf-8')
    result = []
    for line in f2.readlines():
        G = nx.DiGraph()
        index = int(line.strip().split()[0])
        nodes = line.strip().split()[1:]
        G.add_nodes_from(nodes)
        for item in f1.readlines():
            node1, node2 = item.strip().split()
            if (node1 in nodes) and (node2 in nodes):
                G.add_edge(node1, node2)
        degree = nx.degree(G)
        degree = sorted(degree, key=lambda item: item[1], reverse=True)
        num = math.ceil(nodeNum.get(index))
        selectNodes = []
        for node in degree:
            selectNodes.append(node[0])
        selectNodes = selectNodes[0: num]
        result = result + selectNodes
        f3.write(" ".join([str(x) for x in selectNodes]) + '\n')
    f1.close()
    f2.close()
    f3.close()
    return result

# 找到所有长度为L的路径
def forward(Q, D, spd, pp, W, Len, selectedNodes):
    x = Q[-1]
    L = 0

    while L <= Len and (y not in Q) and (y in W):
        Q.append(y)
        pp = pp.b_xy
        spd = spd + pp
        D[x].append(y)
        x = Q[-1]
    return Q, D, spd, pp

"""
计算节点u的影响力
:param u: 候选集中的一个节点
:param w: 输入网络的一部分, V-seeds
:param L: 路径长度
"""
def backtrack(u, W, L, selectedNodes):
    # Q: 节点候选集
    Q = []
    spd = 1
    pp = 1
    # 当前节点的邻居
    D = []

    while not Q:
        Q, D, spd, pp = forward(Q, D, spd, pp, W, L, selectedNodes)
        # Q.pop(): 删除并返回最后一个元素
        u = Q.pop()
        # 删除D[u]

        v = Q[-1]
        pp = pp/b_vu
    return spd

# 获取节点n阶的所有邻居
def get_neigbors(g, node, depth=1):
    result = set()
    output = {}  # {1: [], n: []} 每一阶邻居的结果
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]

    for i in range(1, depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]

    for k, v in output.items():
        result = set.union(result, set(v))
    return result

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
    # 加载整个图
    graph = nx.DiGraph()
    nodes = [x for x in range(args.nodeNum)]
    graph.add_nodes_from(nodes)
    f = open(os.path.dirname(os.path.abspath(__file__)) + '/similarity/' + args.dataset + '_cosine.txt', 'r')
    for line in f.readlines():
        node1, node2, weight = line.strip().split(' ')
        graph.add_edge(int(node1), int(node2), weight=float(weight))
        
    f.close()
    
    communities = utils.slpa(graph, 0.5, 20)
    # 社区图
    G = createCommNet(graph, communities)
    comm_btn = topNBetweeness(G)
    print(comm_btn)











    # 社区图中每个节点的扩展度
    G_spread = SimPath.simpath1(G, 0.001)
    # 将ndarray转为dict
    G_spread = dict(enumerate(G_spread.flatten(), 0))
    # 从大到小排序
    G_spread = sorted(G_spread.items(), key=lambda item: item[1], reverse=True)
    seeds = []
    for node in G_spread[0: args.k]:
        seeds.append(node[0])

    print(seeds)
    end = time.time()

    print('dataset: ' + str(args.dataset))
    f = open('incimResult/incim_' + args.dataset + '_' + str(args.k) + '.txt', 'w', encoding='utf-8')
    graph_query = utils.load_DiGraph_query(args.dataset, args.nodeNum, args.n_input)
    spreadSum = LT(graph_query, seeds, mc=10000, method='pp_random') 
    print('k: ' + str(args.k) + '\n')
    print('seeds_list: ' + str(seeds))
    print('spreadSum: ' + str(spreadSum))
    print('Time: ', end - start)
    f.write('k: ' + str(args.k) + '\n')
    f.write('seeds_list: ' + str(seeds) + '\n')
    f.write('spreadSum: ' + str(spreadSum) + '\n')
    f.write('Time：' + str(end - start) + '\n')
    f.close()









