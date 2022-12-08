import networkx as nx
import numpy as np
import heapq
import utils

# 有向图，LT
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
        top_ = list()
        for t in top:
            top_.append(t[1])
        return top_

    def get_gain(self, node):
        return self.nodes_gain[node]

# 找到所有长度为L的路径
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

# 计算节点u的影响力
def backtrack(u, r, W, U, spdW_, graph):
    Q = [u]
    spd = 1
    pp = 1
    D = list()
    for i in range(graph.number_of_nodes()):
        D.append([])

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
    W = set(graph.nodes()) - S
    if U is None or spdW_ is None:
        spdW_ = np.zeros(graph.number_of_nodes())

    for u in S:
        W.add(u)
        # print spdW_[u]
        spread = spread + backtrack(u, r, W, U, spdW_[u], graph)
        # print spdW_[u]
        W.remove(u)
    return spread

# 获取能覆盖所有边的节点
def get_vertex_cover(graph):
    # dv: 存放图中节点的度
    dv = np.zeros(graph.number_of_nodes())
    check_array = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    checked = 0

    for i in range(graph.number_of_nodes()):
        dv[i] = graph.degree(i)

    V = set()
    while checked < graph.number_of_edges():
        s = dv.argmax()
        V.add(s)
        # make sure that never to select this node again
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

    return V

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
        U = V_C & set(graph.predecessors(u))
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
    for node in range(graph.number_of_nodes()):
        celf.put(node, spread[node])
    S = set()
    W = V  # 所有节点
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
                S.add(x)
                W = W - set([x])
                spd = spread[x]
                # print spread[x],simpath_spread(S,r,None,None)
                checked = np.zeros(graph.number_of_nodes())
                celf.remove(x)
                break
            else:
                spread[x] = backtrack(x, r, W, None, None, graph) + spdV_x[x]
                checked[x] = 1
                celf.update(x, spread[x] - spd)
    return S

if __name__ == '__main__':
    dataset = 'acm'
    nodeNum = 3025
    # path = 'similarity/' + dataset + '_cosine.txt'
    # graph = utils.load_weighted_graph(path, nodeNum)

    graph = utils.load_DiGraph_cos(dataset, nodeNum)

    seeds = simpath(graph, 50, 0.001, 7)
    print(seeds)







