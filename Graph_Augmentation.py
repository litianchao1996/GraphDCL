#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
from scipy.stats import ortho_group
from torch_geometric.utils import to_edge_index

import torch

def inList(array, list):
    for element in list:
        if np.array_equal(element, array):
            return True
    return False

def aug_attr_matrix(attr_matrix, M, edge_index):
    M_transpose = torch.transpose(M, 0, 1)
    attr_dim = attr_matrix.size(2)

    aug_matrix = []

    for i in range(attr_dim):
        attr = attr_matrix[:,:,i]
        aug_attr = torch.mm(torch.mm(M_transpose, attr), M)
        aug_edge_attr = torch.zeros(edge_index.shape[1], 1)
        for i in range(len(edge_index.t())):
            edge = edge_index.t()[i]
            x = edge[0]
            y = edge[1]
            aug_edge_attr[i][0] = aug_attr[x][y]
            
        aug_matrix.append(aug_edge_attr)
    return torch.cat(aug_matrix, dim=1)   
    
class Permutation_Aug():
    def __init__(self, X, A, N=2):
        """
        X: the feature matrix n*d
        A: the adjancy matrix n*n
        N: the number aug graphed 
        
        """
        self.X = X
        self.A = A
        self.N = N
        self.num_node = X.shape[0]
    
    
    def generate_N_permutation_matrix(self):
        """
        N permutation matrixs in torch.tensor
        
        """
        num_node = self.num_node
        N = self.N
        
        per_list =[]
        
        i = np.identity(num_node)
            
        count = 0
    
        while count < N:
            i_ma = np.identity(num_node)
            while (i_ma == i).all():
                i_ma = np.random.permutation(i_ma)       #Get an non-identity permutation matrix
            
            if len(per_list) == 0:
                per_list.append(torch.FloatTensor(i_ma))
                count = count + 1
            
            if inList(i_ma, per_list):
                continue
            else:
                per_list.append(torch.FloatTensor(i_ma))
                count = count + 1
            
        return per_list
    
    def N_permutation_aug(self):
        """
        Generate a list of tuples with permuted X, A
        """
        X = self.X
        A = self.A
        
        aug_list = []
        
        per_list = self.generate_N_permutation_matrix()
    
        for item in per_list:
            
            M = item
            
            M_transpose = torch.transpose(M, 0, 1)
            
            X_aug = torch.mm(M_transpose, X)
            A_aug = torch.mm(torch.mm(M_transpose, A), M)
            aug_list.append((X_aug, A_aug, M))
            
        return aug_list
    
class Transformation_Aug():
    
    def __init__(self, X, A, N=2):
        """
        X: the feature matrix n*d
        A: the adjancy matrix n*n
        N: the number aug graphed 
        
        """
        self.X = X
        self.A = A
        self.N = N
        self.num_node = X.shape[0]

    def generate_N_rand_orthonormal_matrix(self):
        """
        N transformation matrixs in torch.tensor
        
        """
        num_node = self.num_node
        N = self.N
        
        orth_list = []
        count = 0
    
        while count < N:
        
            orth_ma = ortho_group.rvs(dim=num_node)
        
            if len(orth_list) == 0:
                orth_list.append(torch.FloatTensor(orth_ma))
                count = count + 1
        
            if inList(orth_ma, orth_list):
                    continue
            else:
                orth_list.append(torch.FloatTensor(orth_ma))
                count = count + 1
            
        return orth_list
    
    def N_transformation_aug(self):
        """
        Generate a list of tuples with transformed X, A
        """
        
        X = self.X
        A = self.A
        
        aug_list = []
        
        per_list = self.generate_N_rand_orthonormal_matrix()
    
        for item in per_list:
            
            M = item
            
            M_transpose = torch.transpose(M, 0, 1)
            
            X_aug = torch.mm(M_transpose, X)
            A_aug = torch.mm(torch.mm(M_transpose, A), M)
            aug_list.append((X_aug, A_aug, M))
            
        return aug_list
