"""
Rahimkhani K, Aleahmad A, Rahgozar M, et al. A fast algorithm for finding most influential people based on the linear threshold model[J]. Expert Systems with Applications, 2015, 42(3): 1353-1361.
"""
import networkx as nx
import SimPath
import argparse
import time
import utils
from LT import LT
import os

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
                    if G.has_edge(comm1, comm2):
                        G[comm1][comm2]["weight"] = G[comm1][comm2]["weight"] + 1
                    else:
                        G.add_edge(comm1, comm2, weight=1)

    for edge in G.edges():
        i = edge[0]
        j = edge[1]
        G[i][j]["weight"] = G[i][j]["weight"] / G.in_degree(j)
    
    return G

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
    f = open(os.path.dirname(os.path.abspath(__file__)) + '/graph/' + args.dataset + '_graph.txt', 'r')
    for line in f.readlines():
        node1, node2 = line.strip().split(' ')
        graph.add_edge(int(node1), int(node2))
        
    f.close()
    
    communities = utils.slpa(graph, 0.5, 20)
    print(len(communities))
    # 社区图
    G = createCommNet(graph, communities)

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
    f = open('incimResult/incim_' + args.dataset + '_' + str(args.k) + '.txt', 'a', encoding='utf-8')
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




    













