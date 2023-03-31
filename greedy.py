import networkx as nx
import time
from IC_query import IC
import argparse
import numpy 
import utils
import os

"""
  贪心算法
  input: 图、种子数量、传播概率、蒙特卡洛模拟次数
  output: 最佳种子集、扩展度结果、每次迭代时间
"""
def greedy(g, k, p=0.1, mc=1000, method='random'):
    S, spread, timelapse, start_time = [], [], [], time.time()
    # 寻找具有最大边际增益的k个节点
    for _ in range(k):
        best_spread = 0
        # 遍历不在种子集中的节点，找到有最大边际效益的节点
        for j in g.nodes() - set(S):
            # 计算扩展度
            s = IC(g, S+[j], p, mc, method=method)
            if s > best_spread:
                best_spread, node = s, j
        
        S.append(node)
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)
    return (S, spread, timelapse)

if __name__ == "__main__":
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

    # 随机生成维数为属性维数的query
    query = numpy.loadtxt('query/' + args.name + '_query.txt', delimiter=',')
    args.query = query
    G = utils.load_graph_query(args.dataset, args.nodeNum, args.query)

    seeds, SPREAD, timelapse = greedy(G, args.k, mc=10000, method='pp_random')
    spreadSum = SPREAD[-1]
    time = timelapse[-1]

    print('dataset: ' + str(args.dataset))
    f = open('greedyResult/greedy_' + args.dataset + '_' + str(args.k) + '.txt', 'w', encoding='utf-8')
    print('k: ' + str(args.k) + '\n')
    print('seeds_list: ' + str(seeds))
    print('spreadSum: ' + str(spreadSum))
    print('Time: ', time)
    f.write('k: ' + str(args.k) + '\n')
    f.write('seeds_list: ' + str(seeds) + '\n')
    f.write('spreadSum: ' + str(spreadSum) + '\n')
    f.write('Time：' + str(time) + '\n')
    f.close()

    

