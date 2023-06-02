import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class GCNConv(Module):  # 节点聚合层
    def __init__(self,in_features, out_features, bias=True):
        super(GCNConv,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # input_features, out_features 全是0的一个参数矩阵
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  # 随机化参数
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_x, adj):
        support = torch.mm(input_x, self.weight)  # GraphConvolution forward。input*weight
        output = torch.spmm(adj, support)  # 稀疏矩阵的相乘，和mm一样的效果
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class EdgeGCNConv(Module): # 链路聚合层
    def __init__(self, in_features, out_features, bias=True):
        super(EdgeGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # input_features, out_features 全是0的一个参数矩阵
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  # 随机化参数
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_y, inci):
        support = torch.mm(input_y, self.weight)  # GraphConvolution forward。input*weight
        output = torch.mm(inci, support)  # 稀疏矩阵的相乘，和mm一样的效果
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
