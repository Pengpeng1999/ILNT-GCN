import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, edges_GraphConvolution
import torch

class Line_GCN(nn.Module):
    def __init__(self, nfeat, efeat, nhid, nclass, dropout):
        super(Line_GCN, self).__init__()

        self.gc1_vertex = GraphConvolution(nfeat, nhid) # 构建第一层 GCN
        self.gc1_edge = edges_GraphConvolution(efeat, nhid)
        self.gc2 = GraphConvolution(2 * nhid, nclass) # 构建第二层 GCN
        self.dropout = dropout

    def forward(self, x, adj, y ,inc):
        x = F.relu(self.gc1_vertex(x, adj)) # 聚合邻居节点
        y = F.relu(self.gc1_edge(y, inc)) # 聚合关联边
        self.edge_inform = y
        x = F.dropout(x, self.dropout, training=self.training)
        y = F.dropout(y, self.dropout, training=self.training)
        x = torch.cat((x,y), dim =1)
        x = self.gc2(x, adj)
        self.res_x = F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1)  # 将特征矩阵的每一个维度以一个节点为中心，进行softmax

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid) # 构建第一层 GCN
        self.gc2 = GraphConvolution(nhid, nclass) # 构建第二层 GCN
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)   # 将特征矩阵的每一个维度以一个节点为中心，进行softmax

