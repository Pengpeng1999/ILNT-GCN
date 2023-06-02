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
import scipy.sparse as sp
from node2vec import Node2Vec

from utils import load_data, accuracy, normalize, dict_to_array, mm, Data
from models import Line_GCN
from tqdm import tqdm
# function
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad() # GraphConvolution forward
    output = model(features, adj, edges_feature, incidence_matrix)   # 运行模型，输入参数 (features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj, edges_feature, incidence_matrix)

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
    t = time.time()
    model.eval()
    output = model(features, adj, edges_feature, incidence_matrix)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    print('test_time: {:.4f}s'.format(time.time() - t))
    return acc_test.item()

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,  # cora0.01 # citeseer0.001
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
    
# Load data
# adj, features, labels, idx_train, idx_val, idx_test, cora_graph, incidence_matrix, line_cora_graph = load_data()
features, labels, cora_graph= load_data()
'需要x_feature, labels, graph。'


cora_data = Data(cora_graph, features, labels)
x = cora_data.x
embedding_edge_attr = cora_data.embedding_edge_attr
adj = cora_data.adj
incidence_matrix = cora_data.incidence_matrix
labels = cora_data.labels
idx_train = cora_data.idx_train
idx_val = cora_data.idx_val
idx_test = cora_data.idx_test

model = Line_GCN(cora_data.x.shape[1],cora_data.embedding_edge_attr.shape[1],args.hidden,labels.max().item() + 1,args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
if args.cuda:
    model.cuda()
    x = x.cuda()
    embedding_edge_attr = embedding_edge_attr.cuda()
    adj = adj.cuda()
    incidence_matrix = incidence_matrix.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad() # GraphConvolution forward
    output = model(x, adj, embedding_edge_attr, incidence_matrix)   # 运行模型，输入参数 (features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        model.eval()
        output = model(x, adj, embedding_edge_attr, incidence_matrix)

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
    t = time.time()
    output = model(x, adj, embedding_edge_attr, incidence_matrix)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    print('test_time: {:.4f}s'.format(time.time() - t))
    return acc_test.item()

# Train model

t_total = time.time()
for epoch in range(args.epochs):
    a = train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# Testing
test()
