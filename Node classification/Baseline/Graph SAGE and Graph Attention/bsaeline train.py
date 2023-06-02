from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx

from utils import load_data, Data, accuracy, load_pubmed
from baseline_model import SAGE_Net, GAT_Net


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
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

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# MAC: option + command + <-
# Load data
features, labels, graph = load_data()
_data = Data(graph,features, labels)
features = _data.x
labels = _data.labels
idx_train = _data.idx_train
idx_val = _data.idx_val
idx_test = _data.idx_test
idx_nodes = list(graph.nodes)
nodes_idx = {j:i for i,j in enumerate(idx_nodes)}
head_index = []
tail_index = []
for i in graph.edges:
    head_index.append(nodes_idx[i[0]])
    head_index.append(nodes_idx[i[1]])
    tail_index.append(nodes_idx[i[1]])
    tail_index.append(nodes_idx[i[0]])
edges_index = torch.LongTensor([head_index,tail_index])
# Model and optimizer,构造GCN，初始化参数。两层GCN
model = GAT_Net(features.shape[1],
            args.hidden,
            labels.max().item() + 1,)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    edges_index = edges_index.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad() # GraphConvolution forward
    output = model(features, edges_index)   # 运行模型，输入参数 (features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, edges_index)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return output

def test():
    model.eval()
    output = model(features, edges_index)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    a = train(epoch)
print(a.size())
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# Testing
test()
