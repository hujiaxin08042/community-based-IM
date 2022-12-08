import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset
import networkx as nx
from networkx.algorithms import community
import os


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
    return G

# 使用louvain算法划分社区
def louvain(G):
    communities = community.louvain_communities(G)
    return communities