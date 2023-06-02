import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class SAGE_Net(torch.nn.Module):
    def __init__(self,data,dropout = 0.5):
        super(SAGE_Net,self).__init__()
        self.dropout = dropout
        self.data = data
        self.conv1 = SAGEConv(data.x.shape[1], 128)
        self.conv2 = SAGEConv(128, 64)

    def encode(self):
        x = self.conv1(self.data.x, self.data.train_pos_edge_index)
        x = x.relu()
        x = F.dropout(x, self.dropout, training=self.training)
        return self.conv2(x, self.data.train_pos_edge_index)
    
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class GAT_Net(torch.nn.Module):
    def __init__(self,data,dropout = 0.5):
        super(GAT_Net,self).__init__()
        self.dropout = dropout
        self.data = data
        self.conv1 = GATConv(data.x.shape[1], 128)
        self.conv2 = GATConv(128, 64)
    
    def encode(self):
        x = self.conv1(self.data.x, self.data.train_pos_edge_index)
        x = x.relu()
        x = F.dropout(x, self.dropout, training=self.training)
        return self.conv2(x, self.data.train_pos_edge_index)
    
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


