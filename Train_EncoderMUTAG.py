import math

import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from torch_geometric.nn.aggr.lstm import LSTMAggregation
import sys
sys.path.append("/mnt/server-home/TUE/ypei1/Graph_Aug/")

from Graph_Encoder import Node_Aggregator, Graph_Encoder, Node_Encoder, to_adj_matrix, to_index_attr, to_attr_matrix

from Graph_Augmentation import Permutation_Aug, Transformation_Aug, aug_attr_matrix

import logging
import os

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('eval.log', mode='w'),
        stream_handler
    ])

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
logger.info(device)

def loss_cl(x1, x2):
    T = 0.1
    batch_size, _ = x1.size()
    
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    
    loss = - torch.log(loss).mean()
    return loss

def loss_fn(pz1, pz2, tz1, tz2, alpha):
    
    p_loss = loss_cl(pz1, pz2)
    t_loss = loss_cl(tz1, tz2)
    
    loss = alpha * p_loss + (1 - alpha) * t_loss

    return loss

def train_encoder(encoder, optimizer, loss_fn, alpha, num_epochs, data_loader, batch_size, left, GCN=False):
    
    result_list = []
    if left:
        total_step = len(data_loader)-1
    else:
        total_step = len(data_loader)
    for epoch in range(num_epochs):
        
        encoder.train()
        loss = 0
        for step, batch in enumerate(data_loader):
            encoder.to(device)
            p_z1_list = []
            p_z2_list = []
            t_z1_list = []
            t_z2_list = []
            
            if step >= total_step:
                break
            else:
                for i in range(batch_size):
                    node_feats = batch[i].x.cpu()
                    edge_index = batch[i].edge_index.cpu()
                    adj_matrix = to_adj_matrix(node_feats, edge_index).cpu()
                    edge_attr = batch[i].edge_attr.cpu()
                    attr_matrix = to_attr_matrix(node_feats, edge_index, edge_attr)
                    
                    p_list = Permutation_Aug(node_feats, adj_matrix).N_permutation_aug()
                    t_list = Transformation_Aug(node_feats, adj_matrix).N_transformation_aug()
                    #get embeddings for the permutation view
                    p_x1, p_adj1, p_M1 = p_list[0]
                    p_x2, p_adj2, p_M2 = p_list[1]

                    p_edge_index1, p_edge_attr1 = to_index_attr(p_adj1)
                    p_edge_index2, p_edge_attr2 = to_index_attr(p_adj2)
                    p_edge_attr1 = p_edge_attr1.reshape(p_edge_attr1.shape[0], -1)
                    p_edge_attr2 = p_edge_attr2.reshape(p_edge_attr2.shape[0], -1)

                    p_attr1 = aug_attr_matrix(attr_matrix, p_M1, p_edge_index1)
                    p_attr2 = aug_attr_matrix(attr_matrix, p_M2, p_edge_index2)
                    
                    p_edge_attr1 = torch.cat([p_edge_attr1, p_attr1], dim=1)
                    p_edge_attr2 = torch.cat([p_edge_attr2, p_attr2], dim=1)

                    p_edge_attr1, p_edge_attr2 = p_edge_attr1.to(device), p_edge_attr2.to(device)
                    p_x1, p_x2, p_adj1, p_adj2 = p_x1.to(device), p_x2.to(device), p_adj1.to(device), p_adj2.to(device)
                    p_edge_index1, p_edge_index2, p_edge_attr1, p_edge_attr2 = p_edge_index1.to(device), p_edge_index2.to(device), p_edge_attr1.to(device), p_edge_attr2.to(device)
                    if not GCN:
                        p_z1 = encoder(p_x1, p_edge_index1, p_edge_attr1).to(device)
                        p_z1_list.append(p_z1)
                   
                        p_z2 = encoder(p_x2, p_edge_index2, p_edge_attr2).to(device)
                        p_z2_list.append(p_z2)
                    else:
                        p_z1 = encoder(p_x1, p_adj1).to(device)
                        p_z1_list.append(p_z1)
                   
                        p_z2 = encoder(p_x2, p_adj2).to(device)
                        p_z2_list.append(p_z2)

                    #get embeddings for the orthnormal view
                    t_x1, t_adj1, t_M1 = t_list[0]
                    t_x2, t_adj2, t_M2 = t_list[1]
                    

                    t_edge_index1, t_edge_attr1 = to_index_attr(t_adj1)
                    t_edge_index2, t_edge_attr2 = to_index_attr(t_adj2)
                    t_edge_attr1 = t_edge_attr1.reshape(t_edge_attr1.shape[0], -1)
                    t_edge_attr2 = t_edge_attr2.reshape(t_edge_attr2.shape[0], -1)

                    t_attr1 = aug_attr_matrix(attr_matrix, t_M1, t_edge_index1)
                    t_attr2 = aug_attr_matrix(attr_matrix, t_M2, t_edge_index2)
                
                    t_edge_attr1 = torch.cat([t_edge_attr1, t_attr1], dim=1)
                    t_edge_attr2 = torch.cat([t_edge_attr2, t_attr2], dim=1)

                    #t_M1, t_M2 = t_M1.to(device), t_M2.to(device)
                    t_edge_attr1, t_edge_attr2 = t_edge_attr1.to(device), t_edge_attr2.to(device)
                    t_x1, t_x2, t_adj1, t_adj2 = t_x1.to(device), t_x2.to(device), t_adj1.to(device), t_adj2.to(device)
                    t_edge_index1, t_edge_index2, t_edge_attr1, t_edge_attr2 = t_edge_index1.to(device), t_edge_index2.to(device), t_edge_attr1.to(device), t_edge_attr2.to(device)
                    if not GCN:
                        t_z1 = encoder(t_x1, t_edge_index1, t_edge_attr1).to(device)
                        t_z1_list.append(t_z1)
                   
                        t_z2 = encoder(t_x2, t_edge_index2, t_edge_attr2).to(device)
                        t_z2_list.append(t_z2)
                    else:
                        t_z1 = encoder(t_x1, t_adj1, None).to(device)
                        t_z1_list.append(p_z1)
                   
                        t_z2 = encoder(t_x2, t_adj2, None).to(device)
                        t_z2_list.append(p_z2)                                           
                 
            pz1 = torch.cat(p_z1_list, dim=0).to(device)
            pz2 = torch.cat(p_z2_list, dim=0).to(device)
            
            tz1 = torch.cat(t_z1_list, dim=0).to(device)
            tz2 = torch.cat(t_z2_list, dim=0).to(device)
            
            l = loss_fn(pz1, pz2, tz1, tz2, alpha).to(device)
                       
            # Backward and optimize
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            
            loss += l.item()
        
        avg_loss = loss/total_step
        result_list.append(avg_loss)
        output = 'Epoch [{}/{}], dual contrast loss : {:.12f}'
        logger.info(output.format(epoch+1, num_epochs, avg_loss))
        model_save_name = 'alpha10ginelstmMUTAG.pt'
        path = F"./save/{model_save_name}" 
        torch.save(encoder, path)
    return result_list

if __name__ == "__main__":
    dataset = TUDataset('.data', name='MUTAG')

    batch_size = 188
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers = 10, shuffle=True)
    left = len(dataset)%batch_size!=0
    GCN = False

    node_encoder = Node_Encoder(7, 256, 'gine', 5)
    aggregator = Node_Aggregator(256, 1024, 'lstm')
    graph_encoder = Graph_Encoder(1024, 300, node_encoder, aggregator)
    #model_save_name = 'alpha10ginelstmMUTAG.pt'
    #path = F"./save/{model_save_name}" 
    #graph_encoder = torch.load(path)
    
    num_epochs = 500
    alpha = 1
    optimizer = torch.optim.Adam(graph_encoder.parameters(), lr=0.00001)

    loss_list = train_encoder(graph_encoder, optimizer, loss_fn, alpha, num_epochs, data_loader, batch_size, left, GCN)
