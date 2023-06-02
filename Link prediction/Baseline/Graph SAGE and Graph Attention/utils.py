import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


'''def load_data(path="../data/citeseer/", dataset="citeseer"):
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
    features = [dict_features[i] for i in graph.nodes]  # 取特征feature
    labels = np.array([dict_label[i] for i in graph.nodes])
    return features, labels, graph
'''
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
    head = [i[0] for i in list(graph.edges)]
    tail = [i[1] for i in list(graph.edges)]
    adj = sp.coo_matrix((np.ones(len(graph.edges)), (np.array([nodes_idx[i] for i in head]), np.array([nodes_idx[i] for i in tail]))),  # 构建边的邻接矩阵
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix，计算转置矩阵。将有向图转成无向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)   # 对特征做了归一化的操作
    adj = normalize(adj + sp.eye(adj.shape[0]))   # 对A+I归一化
    # 训练，验证，测试的样本
    idx_train = range(int(labels.shape[0] * 0.8))
    idx_val = range(int(labels.shape[0] * 0.8),int(labels.shape[0] * 0.9))
    idx_test = range(int(labels.shape[0] * 0.9), int(labels.shape[0]))
    # 将numpy的数据转换成torch格式
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, graph


def load_data(path="../data/cora/", dataset="cora"):  # citeseer
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
    head = [i[0] for i in list(graph.edges)]
    tail = [i[1] for i in list(graph.edges)]
    adj = sp.coo_matrix((np.ones(len(graph.edges)), (np.array([nodes_idx[i] for i in head]), np.array([nodes_idx[i] for i in tail]))),  # 构建边的邻接矩阵
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix，计算转置矩阵。将有向图转成无向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)   # 对特征做了归一化的操作
    adj = normalize(adj + sp.eye(adj.shape[0]))   # 对A+I归一化
    # 训练，验证，测试的样本
    idx_train = range(int(labels.shape[0] * 0.8))
    idx_val = range(int(labels.shape[0] * 0.8),int(labels.shape[0] * 0.9))
    idx_test = range(int(labels.shape[0] * 0.9), int(labels.shape[0]))
    # 将numpy的数据转换成torch格式
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, graph


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  #  矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def dict_to_array(dict1,edges):
    res = []
    for i in edges:
        if i in dict1.keys():
            res.append(dict1[i])
        elif (i[1],i[0]) in dict1.keys():
            res.append(dict1[(i[1],i[0])])
        else:
            res.append([0 for i in range(len(res[0]))])
    return torch.FloatTensor(res)
