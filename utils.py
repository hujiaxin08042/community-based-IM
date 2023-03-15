import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset
import networkx as nx
from networkx.algorithms import community
import os
import collections

def load_graph(dataset):
    
    abspath = os.path.dirname(os.path.abspath(__file__))
    path = abspath + '/graph/{}_graph.txt'.format(dataset)
    data = np.loadtxt(abspath + '/data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class load_data(Dataset):
    def __init__(self, dataset):
        path = os.path.dirname(os.path.abspath(__file__))
        self.x = np.loadtxt(path + '/data/{}.txt'.format(dataset), dtype=float)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(idx))

def get_graph(path):
    f = open(path, 'r', encoding='utf-8')
    source = []
    target = []
    vertices = set()
    for line in f.readlines():
        n1 = int(line.split(' ')[0])
        n2 = int(line.split(' ')[1].replace('\n', ''))
        vertices.add(n1)
        vertices.add(n2)
        source.append(n1)
        target.append(n2)

    g = nx.DiGraph()
    g.add_nodes_from(vertices)
    g.add_edges_from(zip(source, target))
    return g

# 加载权重为相似度的网络
def load_graph_cos(dataset, nodeNum):
    G = nx.Graph()
    nodes = [x for x in range(nodeNum)]
    G.add_nodes_from(nodes)
    path = os.path.dirname(os.path.abspath(__file__))
    f = open(path + '/similarity/' + dataset + '_cosine.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        n1, n2, weight = line.strip().split()
        G.add_edge(int(n1), int(n2), weight=float(weight))
    return G

# 加载权重为相似度的有向网络
def load_DiGraph_cos(dataset, nodeNum):
    G = nx.DiGraph()
    nodes = [x for x in range(nodeNum)]
    G.add_nodes_from(nodes)
    path = os.path.dirname(os.path.abspath(__file__))
    f = open(path + '/similarity/' + dataset + '_cosine.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        n1, n2, weight = line.strip().split()
        G.add_edge(int(n1), int(n2), weight=float(weight))
        G.add_edge(int(n2), int(n1), weight=float(weight))
    return G

# 使用louvain算法划分社区
def louvain(G):
    communities = community.louvain_communities(G)
    return communities

# 构建社区图，网络中的每个点表示一个社区
def createCommNet():
    f1 = open('../data/NetHEPT_graph.txt', 'r', encoding='utf-8')
    f2 = open('data/NetHEPT_slpa.txt', 'r', encoding='utf-8')
    G = nx.DiGraph()
    node2comm = {}
    count = 0
    for i, line in enumerate(f2):
        count += 1
        for node in line.strip().split(' '):
            if node in node2comm:
                v = node2comm.get(node)
                v.append(i)
                node2comm[node] = v
            else:
                node2comm[node] = [i]

    G.add_nodes_from([i for i in range(count)])

    for line in f1.readlines():
        node1, node2 = line.strip().split(' ')
        comm1_list = node2comm.get(node1)
        comm2_list = node2comm.get(node2)
        if comm1_list and comm2_list and comm1_list != comm2_list:
            for comm1 in comm1_list:
                for comm2 in comm2_list:
                    G.add_edge(comm1, comm2)
    f1.close()
    f2.close()
    return G

# 计算属性相似度和社区与Query之间的相似度，合成影响概率
# 加载权重为影响概率的网络
def load_graph_pp(dataset, nodeNum):
    G = nx.Graph()
    nodes = [x for x in range(nodeNum)]
    G.add_nodes_from(nodes)
    path = os.path.dirname(os.path.abspath(__file__))
    f = open(path + '/similarity/' + dataset + '_pp.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        n1, n2, weight = line.strip().split()
        G.add_edge(int(n1), int(n2), weight=float(weight))
    return G


def load_graph_query(dataset, nodeNum, n_input):
    path = os.path.dirname(os.path.abspath(__file__))
    # 随机生成维数为属性维数的query
    query = np.random.dirichlet(np.ones(n_input), size=1).reshape(n_input,)
    # 计算每一个节点与Query的相似度
    features = np.loadtxt(path + '/data/' + dataset + '.txt', dtype=float)
    node_query_sim = [feature.dot(query) / (np.linalg.norm(feature) * np.linalg.norm(query)) for feature in features]

    G = nx.Graph()
    for i in range(nodeNum):
        G.add_node(i, node_query_sim=node_query_sim[i])
    path = os.path.dirname(os.path.abspath(__file__))
    f = open(path + '/similarity/' + dataset + '_cosine.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        n1, n2, weight = line.strip().split()
        G.add_edge(int(n1), int(n2), weight=float(weight))
    return G


def load_DiGraph_query(dataset, nodeNum, n_input):
    path = os.path.dirname(os.path.abspath(__file__))
    # 随机生成维数为属性维数的query
    query = np.random.dirichlet(np.ones(n_input), size=1).reshape(n_input,)
    # 计算每一个节点与Query的相似度
    features = np.loadtxt(path + '/data/' + dataset + '.txt', dtype=float)
    node_query_sim = [feature.dot(query) / (np.linalg.norm(feature) * np.linalg.norm(query)) for feature in features]

    G = nx.DiGraph()
    for i in range(nodeNum):
        G.add_node(i, node_query_sim=node_query_sim[i])
    
    f = open(path + '/similarity/' + dataset + '_cosine.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        n1, n2, weight = line.strip().split()
        G.add_edge(int(n1), int(n2), weight=float(weight))
        G.add_edge(int(n2), int(n1), weight=float(weight))

    return G


"""
:param path: 路径
:param threshold: 阈值
:param iteration: 迭代次数
https://blog.csdn.net/DreamHome_S/article/details/90113971
"""
def slpa(graph, threshold, iteration):

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

    communities = collections.defaultdict(lambda: list())
    for index, change in enumerate(node_memory):
        for label in change.keys():
            communities[label].append(index)

    res = []
    for community in communities.values():
        res.append([x for x in community])

    return res
