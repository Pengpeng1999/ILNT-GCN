import torch
from layers import GCNConv, EdgeGCNConv
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self,nfeat,efeat,hid_size,out_size, dropout = 0.5):
        super(Net, self).__init__()
        self.dropout = dropout
        self.nfeat = nfeat
        self.efeat = efeat
        self.hid_size = hid_size
        self.out_size = out_size
        self.node_conv1 = GCNConv(nfeat, self.hid_size)
        self.edge_conv1 = EdgeGCNConv(efeat,self.hid_size)
        self.conv2 = GCNConv(2 * self.hid_size, self.out_size)

    def encode(self, data):
        x = data.x
        y = data.train_edges_features
        adj = data.train_graph_adj
        inc = data.train_graph_inci
        x = F.relu(self.node_conv1(x, adj)) # 聚合邻居节点
        y = F.relu(self.edge_conv1(y, inc)) # 聚合关联边
        x = F.dropout(x, self.dropout, training=self.training)
        y = F.dropout(y, self.dropout, training=self.training)
        x = torch.cat((x,y), dim =1)
        x = self.conv2(x, adj)
        self.res_x = x
#         self.res_x = F.log_softmax(x, dim=1)
#         return F.log_softmax(x, dim=1)  # 将特征矩阵的每一个维度以一个节点为中心        
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    
