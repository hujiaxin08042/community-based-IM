from __future__ import print_function, division
import argparse
import numpy
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
import os
from IC_query import IC
from selectInComm_7 import mpCommAI
import networkx as nx
import time
import utils
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z

class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj)
        h = self.gnn_5((1-sigma)*h + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_sdcn():
    dataset = load_data(args.name)
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)

    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name)
    adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch in range(200):
        if epoch % 1 == 0:
            _, tmp_q, pred, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P

        x_bar, q, pred, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    comm_data = [int(x) for x in res3]
    features = numpy.loadtxt(os.path.dirname(os.path.abspath(__file__)) + '/data/' + args.name + '.txt', dtype=float)
    communities = [[] for i in range(args.n_clusters)]
    querySim = []

    for i, d in enumerate(comm_data):
        communities[d].append(i)

    for i in range(args.n_clusters):
        comm = communities[i]
        if len(comm) > 0:
            data = numpy.asarray([features[i].tolist() for i in comm])
            # 计算属性的平均值
            feature = numpy.average(data, axis=0)
            # 计算属性平均值和Query的余弦相似度
            cosine = feature.dot(args.query) / (numpy.linalg.norm(feature) * numpy.linalg.norm(args.query))
            querySim.append(cosine)
        else:
            querySim.append(0)

    return comm_data, querySim

def im(config, comm_data):
    
    # comm_select_num = [config["c1"], config["c2"], config["c3"], config["c4"]]
    comm_select_num = []
    for i in range(args.n_clusters):
        comm_select_num.append(config["c"+str(i)])
    
    seeds_list = mpCommAI(comm_data, args.n_clusters, args.name, args.nodeNum, args.k, comm_select_num)
    # 创建整个网络的图
    G = utils.load_graph_query(args.name, args.nodeNum, args.query)
    spreadSum = IC(G, seeds_list, mc=10000, method='pp_random')
    session.report({"spreadSum": spreadSum, "seeds_list": seeds_list})

def main(comm_data, querySim):

    print("querySim: ", querySim)
    algo = BayesOptSearch(metric="spreadSum", mode="max", utility_kwargs={"kind": "ei", "kappa": 2.5, "xi": 0.04}, querySim=querySim)
    # algo = BayesOptSearch(metric="spreadSum", mode="max", querySim=querySim)
    # 搜索时限制并发数为8
    algo = ConcurrencyLimiter(algo, max_concurrent=8)

    # num_samples = args.n_clusters * 10
    num_samples = 10

    comm_list = []
    for i in range(args.n_clusters):
        c_x = ("c" + str(i), tune.uniform(0, 1))
        comm_list.append(c_x)

    search_space = dict(comm_list)

    # search_space = {
    #     "n_clusters": 4,
    #     "lr": 0.001,
    #     "c1": tune.uniform(0, 1),
    #     "c2": tune.uniform(0, 1),
    #     "c3": tune.uniform(0, 1),
    #     "c4": tune.uniform(0, 1)
    # }
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(im, comm_data=comm_data),
            resources={"cpu": 2, "gpu": 2}
        ),
        tune_config=tune.TuneConfig(
            metric="spreadSum",
            mode="max",
            search_alg=algo,
            num_samples=num_samples,
            # # scheduler: 调度程序，默认FIFO
            scheduler=ASHAScheduler(),
            # # num_samples: 从超参空间采样的次数，默认为1
            # num_samples=1,
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="spreadSum", mode="max", filter_nan_and_inf=False)
    seeds = best_result.metrics["seeds_list"]
    spreadSum = best_result.metrics["spreadSum"]
    print("final seeds: {}".format(seeds))
    print("final spreadSum: {}".format(spreadSum))
    return seeds, spreadSum

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='dblp')
    # parser.add_argument('--name', type=str, default='acm')
    # parser.add_argument('--name', type=str, default='cora')
    # parser.add_argument('--name', type=str, default='citeseer')
    # parser.add_argument('--name', type=str, default='BlogCatalog')
    # parser.add_argument('--name', type=str, default='Sinanet')
    # parser.add_argument('--name', type=str, default='pubmed')
    # parser.add_argument('--name', type=str, default='wiki')
    parser.add_argument('--k', type=int, default=50) 
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=int, default=2)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = os.path.dirname(os.path.abspath(__file__)) + '/data/{}.pkl'.format(args.name)

    if args.name == 'dblp':
        args.n_input = 334
        args.nodeNum = 4057

    if args.name == 'acm':
        args.n_input = 1870
        args.nodeNum = 3025

    if args.name == 'cora':
        args.n_input = 1433
        args.nodeNum = 2708

    if args.name == 'citeseer':
        args.n_input = 3703
        args.nodeNum = 3327

    if args.name == 'BlogCatalog':
        args.n_input = 39
        args.nodeNum = 10312

    if args.name == 'Sinanet':
        args.n_input = 10
        args.nodeNum = 3490

    if args.name == 'pubmed':
        args.n_input = 500
        args.nodeNum = 19717

    if args.name == 'wiki':
        args.n_input = 4973
        args.nodeNum = 2405

    # torch.cuda.set_device(1)
    torch.cuda.set_device(args.device)

    start = time.time()

    # 随机生成维数为属性维数的query
    query = numpy.random.dirichlet(numpy.ones(args.n_input), size=1).reshape(args.n_input,)
    args.query = query
    # 计算每一个节点与Query的相似度
    # features = numpy.loadtxt(os.path.dirname(os.path.abspath(__file__)) + '/data/' + args.name + '.txt', dtype=float)
    # node_query_sim = [feature.dot(query) / (numpy.linalg.norm(feature) * numpy.linalg.norm(query)) for feature in features]

    comm_data, querySim = train_sdcn()
    seeds, spreadSum = main(comm_data, querySim)
    end = time.time()

    print('dataset: ' + str(args.name))
    f = open('sdcn8Result/sdcn_8_' + args.name + '_' + str(args.k) + '.txt', 'a', encoding='utf-8')
    print('k: ' + str(args.k) + '\n')
    print('seeds_list: ' + str(seeds))
    print('spreadSum: ' + str(spreadSum))
    print('Time: ', end - start)
    f.write('k: ' + str(args.k) + '\n')
    f.write('seeds_list: ' + str(seeds) + '\n')
    f.write('spreadSum: ' + str(spreadSum) + '\n')
    f.write('Time：' + str(end - start) + '\n')
    f.close()


    









                                                 



                                                 