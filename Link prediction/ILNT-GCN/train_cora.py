from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import time
import argparse
import numpy as np
import networkx as nx


import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, normalize, dict_to_array, Data, nor_adj, mm, load_pubmed
from model import Net
from tqdm import tqdm

from deepsnap.graph import Graph
from torch_geometric.utils import train_test_split_edges
from node2vec import Node2Vec
import copy
def train(data):
    model.train()
    neg_edge_index = negative_sampling(
        edge_index = data.train_pos_edge_index, num_nodes = data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
        force_undirected=True,
    )
    optimer.zero_grad()
    model.zero_grad()
    z = model.encode(data)  # forwward
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode(data)
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return perfs

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args(args=[])
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# MAC: option + command + <-
# Load data
features, labels, cora_graph = load_data()
cora_data = Data(cora_graph,features, labels)


G = cora_data.graph
data = Graph(G)
data.x = cora_data.x
data.edge_attr = None
data.graph = cora_graph
data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.05)
data.idx_nodes = list(data.graph.nodes)
data.nodes_idx = {j:i for i,j in enumerate(data.idx_nodes)}
data.train_graph_adj = nor_adj(data.train_pos_edge_index, data.x.shape[0])
train_graph = copy.deepcopy(data.graph)
train_graph.remove_edges_from(data.graph.edges)
train_graph_edges = data.train_pos_edge_index.t().tolist()

for i in range(len(train_graph_edges)):
    for j in range(2):
        train_graph_edges[i][j] = data.idx_nodes[train_graph_edges[i][j]]




train_graph.add_edges_from(train_graph_edges)
data.train_graph = train_graph
data.train_graph_inci = torch.FloatTensor(normalize(nx.incidence_matrix(data.train_graph)).A)
data.line_train_graph = nx.line_graph(data.train_graph)
lone_node = []
for i in data.line_train_graph.nodes():
    if data.line_train_graph.degree(i) == 0:
        lone_node.append(i)
    
# 图嵌入
embedding_size = 128
n2v = Node2Vec(data.line_train_graph, dimensions = embedding_size, p = 1, q = 1, walk_length = 10, num_walks = 80, workers = 4)
model = n2v.fit(window=3,min_count=1)
embeddings = model.wv



import tqdm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score



# 获取节点嵌入向量矩阵并将其转换为一个字典
embed_dict = {}
for node in data.line_train_graph.nodes():
    if node in lone_node:
        embed_dict[node] = [0 for i in range(embedding_size)]
    else:
        embed_dict[node] = embeddings[str(node)].tolist()
# with open(r'embed_dict.txt','w') as f:
#     f.write(str(embed_dict))
# with open(r'embed_dict.txt','r') as f:
#     embed_dict = eval(f.read())
edges_feature = dict_to_array(embed_dict, data.train_graph.edges)
edges_feature = torch.FloatTensor(mm(edges_feature))
data.train_edges_features = edges_feature  # 按照list（train_graph.edges）顺序排列
# setting model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net(data.x.shape[1], embedding_size, 128, 64).to(device), data.to(device)
optimer = torch.optim.Adam(params = model.parameters(),lr = args.lr)   


# train model
best_val_perf = test_perf = 0
for epoch in range(args.epochs):
    train_loss = train(data)
    val_perf, tmp_test_perf = test(data)
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, best_val_perf, test_perf))
