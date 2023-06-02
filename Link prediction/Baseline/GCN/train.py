import networkx as nx
from deepsnap.graph import Graph
import torch

from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch.functional import F
from torch_geometric.utils import train_test_split_edges
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np


import time
import argparse
import numpy as np
import networkx as nx
from model import Net
import torch.optim as optim

from utils import load_data, accuracy, normalize, dict_to_array, load_pubmed

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
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
adj, features, labels, idx_train, idx_val, idx_test, cora_graph = load_data() # Training settings

G = cora_graph
data = Graph(G)
data.x = features
data.edge_attr = None
data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.05)
data.G = cora_graph
data.idx_nodes = list(cora_graph.nodes)
data.nodes_idx = {j:i for i,j in enumerate(data.idx_nodes)}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net(data).to(device), data.to(device)
optimer = torch.optim.Adam(params = model.parameters(),lr = args.lr)



def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train():
    model.train()
    neg_edge_index = negative_sampling(
        edge_index = data.train_pos_edge_index, num_nodes = data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
        force_undirected=True,
    )
    optimer.zero_grad()
    model.zero_grad()
    z = model.encode()  # forward
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode()
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return perfs




best_val_perf = test_perf = 0
for epoch in range(200):
    train_loss = train()
    val_perf, tmp_test_perf = test()
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, best_val_perf, test_perf))

