#!/usr/bin/env python
# coding: utf-8
import math
import numpy as np

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


import torch.nn.functional as F
from torch import nn

from torch_geometric.utils import to_edge_index
from torch_geometric.nn.conv import GINEConv, TransformerConv, WLConvContinuous, NNConv

def to_adj_matrix(feature_matrix, edge_index):
    dimension = feature_matrix.size(0)
    adj_matrix = torch.zeros(dimension, dimension)
    for edge in edge_index.t():
        x = edge[0]
        y = edge[1]
        adj_matrix[x][y] = 1
        adj_matrix[y][x] = 1
    
    return adj_matrix

def to_attr_matrix(feature_matrix, edge_index, edge_attr):
    dimension = feature_matrix.size(0)
    attr_dim = edge_attr.size(1)
    attr_matrix = torch.zeros(dimension, dimension, attr_dim)
    for i in range(len(edge_index.t())):
        edge = edge_index.t()[i]
        x = edge[0]
        y = edge[1]
        for j in range(attr_dim):
            attr_matrix[x][y][j] = edge_attr[i][j]

    return attr_matrix
    
def to_index_attr(adj_matrix):
    edge_index, edge_attr = to_edge_index(adj_matrix.to_sparse())
    #edge_attr = edge_attr.reshape(edge_attr.shape[0], -1)
    return edge_index, edge_attr

class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(GCNLayer, self).__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """Forward.

        Args:
            node_feats: Tensor with node features of shape [ num_nodes, c_in]
            adj_matrix: Adjacency matrices of the graph. If there is an edge from i to j,
                         adj_matrix[i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
                         Assumes to already have added the identity connections.
                         Shape: [num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_nodes, _ = node_feats.size()
        I = torch.eye(num_nodes).to('cuda')
        self_adj = adj_matrix + I
        num_neighbours = self_adj.sum(dim=-1, keepdims=True)   #degree matrix
        node_feats = self.projection(node_feats)
        node_feats = torch.mm(self_adj, node_feats)
        node_feats = node_feats / (num_neighbours)
        return node_feats
    
class GCN(nn.Module):
    def __init__(self, c_in, c_out):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(c_in, c_out)
        self.gcn2 = GCNLayer(c_out, c_out)
        
    def forward(self, node_feats, adj_matrix):
        node_feats = F.relu(self.gcn1(node_feats, adj_matrix))
        node_feats = F.relu(self.gcn2(node_feats, adj_matrix))
        return node_feats

class Graph_Encoder(nn.Module):
    def __init__(self, c_in, c_out, Node_encoder, Aggregator):
        super(Graph_Encoder, self).__init__()
        self.gcn = Node_encoder
        self.agg = Aggregator
        self.projection_head = nn.Sequential(nn.Linear(c_in, c_out), nn.ReLU(inplace=True), nn.Linear(c_out, c_out))
        
    def forward(self, node_feats, edge_index, edge_attr=None, orth_ma=None):
        if edge_attr == None:
            node_emb = self.gcn(node_feats, edge_index)
        else:
            node_emb = self.gcn(node_feats, edge_index, edge_attr)

        if orth_ma != None:
            node_emb = torch.mm(orth_ma, node_emb)
        graph_emb = self.agg(node_emb)
        rep = self.projection_head(graph_emb)
        return rep

    def forward_emb(self, node_feats, edge_index, edge_attr=None, orth_ma=None):
        if edge_attr == None:
            node_emb = self.gcn(node_feats, edge_index)
        else:
            node_emb = self.gcn(node_feats, edge_index, edge_attr)

        if orth_ma != None:
            node_emb = torch.mm(orth_ma, node_emb)
        graph_emb = self.agg(node_emb)
        
        return graph_emb
    
    def get_embeddings(self, loader):

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embs = []
        y = []
        with torch.no_grad():
            for data in loader:
                for i in range(len(data)):
                    x, edge_index= data[i].x, data[i].edge_index
                    #if x is None:
                        #x = torch.ones((batch.shape[0],1))
                    
                    emb = self.forward_emb(x, edge_index)
                    embs.append(emb.cpu().numpy())
                    y.append(data[i].y.cpu().numpy())

        embs = np.concatenate(embs, 0)
        y = np.concatenate(y, 0)
       
        return embs, y
    
class Node_Encoder(nn.Module):
    def __init__(self, c_in, c_out, gcn_layer, edge_dim=None):
        """
        gcn_layer: GATconv, GCNconv, GINEconv, TransformerConv, NNConv, WLConvContinuous
        """
        super(Node_Encoder, self).__init__()
        if edge_dim == None:
            # GATConv, GCNConv, WLConvContinuous
            if  gcn_layer == 'gat':
                from torch_geometric.nn.conv import GATConv
                self.gcn1 = GATConv(c_in, c_out)
                self.gcn2 = GATConv(c_out, c_out)
            elif gcn_layer == 'gcn':
                self.gcn1 = GCNLayer(c_in, c_out)
                self.gcn2 = GCNLayer(c_out, c_out)
            elif gcn_layer == 'wlc':
                self.gcn1 = WLconv(c_in, c_out)
                self.gcn2 = WLconv(c_out, c_out)
        else:
            # GINEConv, TransformerConv, NNConv
            if gcn_layer == 'gine':
                self.gcn1 = GINEconv(c_in, c_out, edge_dim)
                self.gcn2 = GINEconv(c_out, c_out, edge_dim)
            elif gcn_layer == 'trans':
                self.gcn1 = Transformerconv(c_in, c_out, edge_dim)
                self.gcn2 = Transformerconv(c_out, c_out, edge_dim) 
            elif gcn_layer == 'nnconv':
                self.gcn1 = NNconv(c_in, c_out, edge_dim)
                self.gcn2 = NNconv(c_out, c_out, edge_dim)
                
    def forward(self, node_feats, edge_index, edge_attr=None):
        if edge_attr == None:
            node_feats = F.relu(self.gcn1(node_feats, edge_index))
            node_feats = F.relu(self.gcn2(node_feats, edge_index))
        else:
            edge_attr = edge_attr.reshape(edge_index.shape[1], -1)
            node_feats = F.relu(self.gcn1(node_feats, edge_index, edge_attr))
            node_feats = F.relu(self.gcn2(node_feats, edge_index, edge_attr))
        return node_feats
    
class GINEconv(nn.Module):
    def __init__(self, c_in, c_out, edge_dim):
        super(GINEconv, self).__init__()
        self.mlp = torch.nn.Sequential(
                                       nn.Linear(c_in, 2*c_out), 
                                       nn.ReLU(), 
                                       nn.Linear(2*c_out, c_out))
        self.gcn = GINEConv(self.mlp)
        self.edge = nn.Linear(edge_dim, c_in)
        
    def forward(self, node_feats, edge_index, edge_attr=None):
        if edge_attr == None:
            edge_attr = torch.zeros(edge_index.shape[1], node_feats.shape[1])
        else:
            edge_attr = self.edge(edge_attr)
        
        out = self.gcn(node_feats, edge_index, edge_attr)
        
        return out
    
class Transformerconv(nn.Module):
    def __init__(self, c_in, c_out, edge_dim):
        super(Transformerconv, self).__init__()
        self.edge_dim = edge_dim
        self.gcn = TransformerConv(c_in, c_out, edge_dim=edge_dim)

        
    def forward(self, node_feats, edge_index, edge_attr=None):
        if edge_attr == None:
            edge_attr = torch.zeros(edge_index.shape[1], self.edge_dim)
        
        out = self.gcn(node_feats, edge_index, edge_attr)
        
        return out
    
class WLconv(nn.Module):
    def __init__(self, c_in, c_out):
        super(WLconv, self).__init__()
       
        self.gcn = WLConvContinuous()
        self.mlp = torch.nn.Sequential(
                                       nn.Linear(c_in, 2*c_out), 
                                       nn.ReLU(), 
                                       nn.Linear(2*c_out, c_out))
        
    def forward(self, node_feats, edge_index, edge_attr=None):
        
        out = self.gcn(node_feats, edge_index, edge_attr)
        out = self.mlp(out)
        
        return out
    
class NNconv(nn.Module):
    def __init__(self, c_in, c_out, edge_dim):
        super(NNconv, self).__init__()
        self.edge_dim = edge_dim
        self.mlp = torch.nn.Sequential(
                                       nn.Linear(self.edge_dim, c_out), 
                                       nn.ReLU(), 
                                       nn.Linear(c_out, c_in*c_out))
        self.gcn = NNConv(c_in, c_out, self.mlp)
        
    def forward(self, node_feats, edge_index, edge_attr=None):
        
        if edge_attr == None:
            edge_attr = torch.zeros(edge_index.shape[1], self.edge_dim)
        else:
            edge_attr = edge_attr
      
        out = self.gcn(node_feats, edge_index, edge_attr)
        
        return out 

class Node_Aggregator(nn.Module):
    def __init__(self, c_in, c_out, aggregator, max_num_node=None):
        """
        aggregator: LSTMAggregation, GRUAggregation, MLPAggregation, LCMAggregation
        """
        super(Node_Aggregator, self).__init__()
        if aggregator == 'lstm':
            from torch_geometric.nn.aggr import LSTMAggregation
            self.aggregator = LSTMAggregation(c_in, c_out)
        elif aggregator == 'gru':
            from torch_geometric.nn.aggr import GRUAggregation
            self.aggregator = GRUAggregation(c_in, c_out)
        elif aggregator == 'mlp':
            from torch_geometric.nn.aggr import MLPAggregation
            self.aggregator = MLPAggregation(c_in, c_out, max_num_node, num_layers=1)
        elif aggregator == 'lcm':
            from torch_geometric.nn.aggr.lcm import LCMAggregation
            self.aggregator = LCMAggregation(c_in, c_out) 
        
    def forward(self, node_feats):
        graph_emb = self.aggregator(node_feats)
        return graph_emb
