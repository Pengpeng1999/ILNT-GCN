import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class SAGE_Net(torch.nn.Module):
    def __init__(self,input_size,hid_size,class_size):
        super(SAGE_Net,self).__init__()
        self.conv1 = SAGEConv(input_size,hid_size)
        self.conv2 = SAGEConv(hid_size,class_size)

    def forward(self,x, edges_index):
        x = self.conv1(x, edges_index)
        x = F.relu(x)
        x = F.dropout(x,training = self.training)
        x = self.conv2(x, edges_index)

        return F.log_softmax(x,dim = 1)

class GAT_Net(torch.nn.Module):
    def __init__(self,input_size,hid_size,class_size):
        super(GAT_Net,self).__init__()
        self.conv1 = GATConv(input_size,hid_size)
        self.conv2 = GATConv(hid_size,class_size)

    def forward(self,x, edges_index):
        x = self.conv1(x, edges_index)
        x = F.relu(x)
        x = F.dropout(x,training = self.training)
        x = self.conv2(x, edges_index)

        return F.log_softmax(x,dim = 1)
