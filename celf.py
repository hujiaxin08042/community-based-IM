import networkx as nx
import time
from IC_query import IC
import argparse
import numpy 
import utils
import os


def celf(g, k, p=0.1, mc=10000, method='random'):
    # --------------------
    # 用贪婪算法找到第一个节点
    # --------------------

    # 计算第一次迭代排序列表
    start_time = time.time()
    # 计算每个节点的扩展度
    marg_gain = [IC(g, [node], p, mc, method=method) for node in g.nodes()]
    # sorted(iterable, key, reverse=False)：排序且不改变可迭代对象本身
    # - iterable: 可迭代对象，如集合、元组、数组
    # - key: 可自定义排序逻辑
    # - reverse：True为降序，False为升序
    # zip()：将对象中的元素打包成一个元组，返回由这些元组组成的列表
    # 创建节点的排序列表及其边际收益
    Q = sorted(zip(g.nodes(), marg_gain), key=lambda x: x[1], reverse=True)

    # 选择第一个节点，并将其从候选列表中删除
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [len(g.nodes())], [time.time() - start_time]

    # --------------------
    # 使用列表排序过程查找剩下的 k-1 个节点
    # --------------------
    for _ in range(k - 1):

        check, node_lookup = False, 0

        # 在每次迭代中，计算顶部节点的扩展度，然后重新排序，重排后顶部节点不变，将该节点选为下一个种子节点，如果不是，则评估列表中新顶部节点的扩展度
        while not check:
            # 统计计算扩展度的次数
            node_lookup += 1
            # 重新计算顶部节点的扩展度
            current = Q[0][0]
            Q[0] = (current, IC(g, S + [current], p, mc, method=method) - spread)
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            # 检查重新排序后顶部节点是否发生变化
            check = (Q[0][0] == current)
        # 选择下一个节点
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        # 从列表中移除选择的节点
        Q = Q[1:]

    return (S, SPREAD, timelapse, LOOKUPS)


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

    G = utils.load_graph_query(args.dataset, args.nodeNum, args.n_input)
    seeds, SPREAD, timelapse, LOOKUPS = celf(G, args.k, mc=10000, method='pp_random')
    spreadSum = SPREAD[-1]
    time = timelapse[-1]

    print('dataset: ' + str(args.dataset))
    f = open('celfResult/celf_' + args.dataset + '_' + str(args.k) + '.txt', 'w', encoding='utf-8')
    print('k: ' + str(args.k) + '\n')
    print('seeds_list: ' + str(seeds))
    print('spreadSum: ' + str(spreadSum))
    print('Time: ', time)
    f.write('k: ' + str(args.k) + '\n')
    f.write('seeds_list: ' + str(seeds) + '\n')
    f.write('spreadSum: ' + str(spreadSum) + '\n')
    f.write('Time：' + str(time) + '\n')
    f.close()

    