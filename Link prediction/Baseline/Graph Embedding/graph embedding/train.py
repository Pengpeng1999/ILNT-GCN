import numpy as np
import pandas as pd
import networkx as nx
import random
import itertools as it
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics, preprocessing
import torch

import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
from deepsnap.graph import Graph
from torch_geometric.utils import train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score

from embedding.models.line import LINE
from embedding.models.deepwalk import DeepWalk
from embedding.models.node2vec import Node2Vec


def mm(data):
    mm = MinMaxScaler()
    res = mm.fit_transform(data)
    return res

class Data():
    "对数据归纳"
    def __init__(self, graph,x, labels):
        self.graph = graph
        self.idx_nodes = list(graph.nodes)
        self.nodes_idx = {j:i for i,j in enumerate(self.idx_nodes)}
        head = [i[0] for i in list(graph.edges)]
        tail = [i[1] for i in list(graph.edges)]
        adj = sp.coo_matrix((np.ones(len(graph.edges)), (np.array([self.nodes_idx[i] for i in head]), np.array([self.nodes_idx[i] for i in tail]))),  # 构建边的邻接矩阵
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix，计算转置矩阵。将有向图转成无向图
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        x = self.mm(x)   # 对特征做了归一化的操作
        adj = normalize(adj + sp.eye(adj.shape[0]))   # 对A+I归一化
        # 训练，验证，测试的样本
        idx_train = range(int(labels.shape[0] * 0.8))
        idx_val = range(int(labels.shape[0] * 0.8),int(labels.shape[0] * 0.9))
        idx_test = range(int(labels.shape[0] * 0.9), int(labels.shape[0]))
        # 将numpy的数据转换成torch格式
        self.x = torch.FloatTensor(x)
        self.labels = torch.LongTensor(np.where(labels)[1])
        self.adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        self.idx_train = torch.LongTensor(idx_train)
        self.idx_val = torch.LongTensor(idx_val)
        self.idx_test = torch.LongTensor(idx_test)
        '''incidence_matrix = normalize(nx.incidence_matrix(graph).A)
        self.incidence_matrix = torch.FloatTensor(incidence_matrix)
        self.line_graph = nx.line_graph(graph)
        self.embedding_edge_attr = self.embedding(self.line_graph,embedding_size = 100)
    
    def embedding(self,line_cora_graph,embedding_size):
        print("Line grpah embedding start!")
        n2v = Node2Vec(line_cora_graph, dimensions=embedding_size,p = 0.5,q = 2,walk_length = 10,num_walks = 80,workers = 4)
        embedding_model = n2v.fit(window=3,min_count=1)
        embeddings = embedding_model.wv
        embed_dict = {}
        for node in line_cora_graph.nodes():
            embed_dict[node]=embeddings[str(node)].tolist()
        edges_feature = dict_to_array(embed_dict,self.graph.edges)
        edges_feature = torch.FloatTensor(self.mm(edges_feature))
        print("Line grpah embedding finishing!")
        return edges_feature'''
  
    def dict_to_array(self,dict1,edges):
        res = []
        for i in edges:
            if i in dict1.keys():
                res.append(dict1[i])
            elif (i[1],i[0]) in dict1.keys():
                res.append(dict1[(i[1],i[0])])
            else:
                res.append([0 for i in range(len(res[0]))])
        return torch.FloatTensor(res)
    
    def normalize(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))  #  矩阵行求和
        r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
        r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
        r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
        mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
        return mx
    
    def mm(self,data):
        minmax = MinMaxScaler()
        res = minmax.fit_transform(data)
        return res
    
    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

def load_data(path="../data/citeseer/", dataset="citeseer"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),dtype=np.dtype(str))
    graph = nx.Graph(edges_unordered.tolist())
    idx = idx_features_labels[:, 0].astype(str).tolist()
    graph.remove_nodes_from(i for i in list(graph.nodes) if i not in idx)
    idx_nodes = list(graph.nodes)
    nodes_idx = {j:i for i,j in enumerate(idx_nodes)}
    dict_features = dict(zip(idx_features_labels[:, 0].tolist(), idx_features_labels[:, 1:-1].astype(float).tolist()))
    dict_label = dict(zip(idx_features_labels[:, 0].tolist(), encode_onehot(idx_features_labels[:, -1]).tolist()))# one-hot label
    features = np.array([dict_features[i] for i in graph.nodes])  # 取特征feature
    labels = np.array([dict_label[i] for i in graph.nodes])
    return features, labels, graph

def load_pubmed(path="../data/Pubmed-Diabetes/", dataset="Pubmed-Diabetes"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = pd.read_csv(path + 'pubmed_feature.csv').values.astype(str)
    edges_unordered = pd.read_csv(path + 'pubmed_edges.csv').values.astype(str)
    graph = nx.Graph(edges_unordered.tolist())
    idx = idx_features_labels[:, 0].astype(str).tolist()
    graph.remove_nodes_from(i for i in list(graph.nodes) if i not in idx)
    idx_nodes = list(graph.nodes)
    nodes_idx = {j:i for i,j in enumerate(idx_nodes)}
    dict_features = dict(zip(idx_features_labels[:, 0].tolist(), idx_features_labels[:, 1:-1].astype(float).tolist()))
    dict_label = dict(zip(idx_features_labels[:, 0].tolist(), encode_onehot(idx_features_labels[:, -1]).tolist()))# one-hot label
    features = np.array([dict_features[i] for i in graph.nodes])  # 取特征feature
    labels = np.array([dict_label[i] for i in graph.nodes])
    return features, labels, graph

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  #  矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx


features, labels, cora = load_cora()
cora.remove_nodes_from(list(nx.isolates(cora)))
data = Graph(cora)
data.x = features
data.edge_attr = None
data = train_test_split_edges(data)
data.G = cora
data.idx_nodes = list(cora.nodes)
data.nodes_idx = {j:i for i,j in enumerate(data.idx_nodes)}
data.neg_edge_index = negative_sampling(
        edge_index = data.edge_label_index, num_nodes = data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1),
        force_undirected=True,
    )
edge_index = torch.cat([data.edge_label_index, data.neg_edge_index], dim=-1)



  # 删除孤立顶点
# embedding model
line_embed_model = LINE(cora,128,5)
line_embed_model.train()
line_feature_embedding = line_embed_model.get_embeddings()

deepwalk_embed_model = DeepWalk(cora,10,80)
deepwalk_embed_model.train()
deepwalk_feature_embedding = deepwalk_embed_model.get_embeddings()

node2vec_embed_model = Node2Vec(cora,10,80,p=2,q=0.5)
node2vec_embed_model.train()
node2vec_feature_embedding = node2vec_embed_model.get_embeddings()

line_X = torch.FloatTensor([line_feature_embedding[i] for i in list(cora.nodes)])
line_X = torch.cat((line_X[edge_index[0]],line_X[edge_index[1]]),dim = 1).numpy()
deepwalk_X = torch.FloatTensor([deepwalk_feature_embedding[i] for i in list(cora.nodes)])
deepwalk_X = torch.cat((deepwalk_X[edge_index[0]],deepwalk_X[edge_index[1]]),dim = 1).numpy()
node2vec_X = torch.FloatTensor([node2vec_feature_embedding[i] for i in list(cora.nodes)])
node2vec_X = torch.cat((node2vec_X[edge_index[0]],node2vec_X[edge_index[1]]),dim = 1).numpy()
list_labels = np.array([1 for i in range(data.edge_label_index.size(1))] + [0 for i in range(data.neg_edge_index.size(1))])



# train
X_train, X_test, y_train, y_test = train_test_split(line_X, list_labels, test_size=0.2)
log_reg = linear_model.LogisticRegression(multi_class='ovr', solver='lbfgs')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
roc = roc_auc_score(y_test, y_pred)
print('Accuracy:', roc)

X_train, X_test, y_train, y_test = train_test_split(deepwalk_X, list_labels, test_size=0.2)
log_reg = linear_model.LogisticRegression(multi_class='ovr', solver='lbfgs')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
roc = roc_auc_score(y_test, y_pred)
print('Accuracy:', roc)

X_train, X_test, y_train, y_test = train_test_split(node2vec_X, list_labels, test_size=0.2)
log_reg = linear_model.LogisticRegression(multi_class='ovr', solver='lbfgs')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
roc = roc_auc_score(y_test, y_pred)
print('Accuracy:', roc)
