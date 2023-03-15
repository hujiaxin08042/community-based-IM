"""
https://blog.csdn.net/DreamHome_S/article/details/90113971
"""
import collections
import networkx as nx
import numpy as np
import os

"""
:param path: 路径
:param threshold: 阈值
:param iteration: 迭代次数
"""
def slpa(path, threshold, iteration, nodeNum):
    graph = nx.DiGraph()
    nodes = [x for x in range(nodeNum)]
    graph.add_nodes_from(nodes)
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            node1, node2 = line.strip().split(' ')
            graph.add_edge(int(node1), int(node2))
    f.close()

    # 给每个节点增加标签
    for node, data in graph.nodes(True):
        data['label'] = node

    # 节点存储器初始化
    node_memory = []
    for n in range(graph.number_of_nodes()):
        node_memory.append({n: 1})

    # 算法迭代过程
    for t in range(iteration):
        # 任意选择一个监听器
        # np.random.permutation(): 随机排序序列
        order = [x for x in np.random.permutation(graph.number_of_nodes())]
        for i in order:
            label_list = {}
            # 从speaker中选择一个标签传播到listener
            for j in graph.neighbors(i):
                sum_label = sum(node_memory[j].values())
                # np.random.multinomial(): 从多项式分布中提取样本
                label = list(node_memory[j].keys())[np.random.multinomial(
                    1, [float(c) / sum_label for c in node_memory[j].values()]).argmax()]
                label_list[label] = label_list.setdefault(label, 0) + 1
            if len(label_list) > 0:
                # listener选择一个最流行的标签添加到内存中
                selected_label = max(label_list, key=label_list.get)
                node_memory[i][selected_label] = node_memory[i].setdefault(selected_label, 0) + 1

    # 根据阈值threshold删除不符合条件的标签
    for memory in node_memory:
        sum_label = sum(memory.values())
        threshold_num = sum_label * threshold
        for k, v in list(memory.items()):
            if v < threshold_num:
                del memory[k]
    # 返回划分结果 {节点: 标签}
    return node_memory

if __name__ == "__main__":
    dataset = 'BlogCatalog'
    nodeNum = 10312
    path = os.path.dirname(os.path.abspath(__file__)) + '/graph/' + dataset + '_graph.txt'
    
    node_memory = slpa(path, 0.5, 20, nodeNum)
    communities = collections.defaultdict(lambda: list())
    
    for index, change in enumerate(node_memory):
        for label in change.keys():
            communities[label].append(index)

    res = []

    for community in communities.values():
        res.append([x for x in community])
    
    print(len(res))





