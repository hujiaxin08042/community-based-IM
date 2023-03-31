import time
import ray
from ray import tune
from ray.air import session
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter
import networkx as nx
import os
import argparse
import numpy 
import utils


# def evaluate(step, width, height):
#     time.sleep(0.1)
#     # 超参数是width和height
#     return (0.1 + width * step / 100) ** (-1) + height * 0.1

# def objective(config):
#     for step in range(config["steps"]):
#         score = evaluate(step, config["width"], config["height"])
#         session.report({"iterations": step, "mean_loss": score})

# algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
# # 搜索时限制并发数为4
# algo = ConcurrencyLimiter(algo, max_concurrent=4)

# num_samples = 10

# search_space = {
#     "steps": 100,
#     "width": tune.uniform(0, 20),
#     "height": tune.uniform(-100, 100),
#     "func": tune.sample_from(lambda spec: spec.search_space.width * spec.search_space.height),
# }

# print(search_space)

# tuner = tune.Tuner(
#     objective,
#     tune_config=tune.TuneConfig(
#         metric="mean_loss",
#         mode="min",
#         search_alg=algo,
#         num_samples=num_samples,
#     ),
#     param_space=search_space,
# )

# results = tuner.fit()

# print("Best hyperparameters found were: ", results.get_best_result().config)

# 计算图的平均度
# def load_graph(dataset, nodeNum):
#     G = nx.Graph()
#     nodes = [x for x in range(nodeNum)]
#     G.add_nodes_from(nodes)
#     path = os.path.dirname(os.path.abspath(__file__))

#     f = open(path + '/graph/' + dataset + '_graph.txt', 'r', encoding='utf-8')
#     for line in f.readlines():
#         n1, n2 = line.strip().split()
#         G.add_edge(int(n1), int(n2))

#     d = dict(nx.degree(G))
#     print(d)
#     print("最大度为：", max(d.values()))
#     print("平均度为：", sum(d.values())/len(G.nodes))

# load_graph('dblp', 4057)
# load_graph('acm', 3025)
# load_graph('cora', 2708)
# load_graph('citeseer', 3327)
# load_graph('BlogCatalog', 10312)
# load_graph('Sinanet', 3490)
# load_graph('pubmed', 19717)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--dataset', type=str, default='dblp')
    # parser.add_argument('--dataset', type=str, default='acm')
    # parser.add_argument('--dataset', type=str, default='cora')
    # parser.add_argument('--dataset', type=str, default='citeseer')
    # parser.add_argument('--dataset', type=str, default='BlogCatalog')
    # parser.add_argument('--dataset', type=str, default='Sinanet')
    # parser.add_argument('--dataset', type=str, default='pubmed')
    # parser.add_argument('--k', type=int, default=10)
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
    query = numpy.random.dirichlet(numpy.ones(args.n_input), size=1).reshape(args.n_input,)
    path = os.path.dirname(os.path.abspath(__file__))
    path = path + '/query/' + args.dataset + '_query.txt'
    numpy.savetxt(path, query)


    

    

