#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from Graph_Encoder import Graph_Encoder, Node_Encoder, Node_Aggregator
import sys
sys.path.append("/mnt/server-home/TUE/ypei1/Graph_Aug/")

import logging
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

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def logistic_classify(x, y):

    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]
    
    accs_train = []
    accs_test = []
    accs_val = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):

        # test
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs), torch.from_numpy(train_lbls)
        test_embs, test_lbls= torch.from_numpy(test_embs), torch.from_numpy(test_lbls)


        log = LogReg(hid_units, nb_classes)
        #log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()
            
        logits = log(train_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == train_lbls).float() / train_lbls.shape[0]
        accs_train.append(acc.item())
        
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs_test.append(acc.item())

        # val
        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs), torch.from_numpy(train_lbls)
        test_embs, test_lbls= torch.from_numpy(test_embs), torch.from_numpy(test_lbls)


        log = LogReg(hid_units, nb_classes)
        
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs_val.append(acc.item())
    #print(len(accs_val), len(accs))
    #return np.mean(accs_val), np.mean(accs)
    
    return accs_train, accs_val, accs_test

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies_train = []
    accuracies_test = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_train.append(accuracy_score(y_train, classifier.predict(x_train)))
        accuracies_test.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    #return np.mean(accuracies_val), np.mean(accuracies)
    return accuracies_train, accuracies_val, accuracies_test

def randomforest_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies_train = []
    accuracies_test = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies_train.append(accuracy_score(y_train, classifier.predict(x_train)))
        accuracies_test.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    #ret = np.mean(accuracies)
    #return np.mean(accuracies_val), ret
    return accuracies_train, accuracies_val, accuracies_test

def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies_train = []
    accuracies_test = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies_train.append(accuracy_score(y_train, classifier.predict(x_train)))
        accuracies_test.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    #return np.mean(accuracies_val), np.mean(accuracies)
    return accuracies_train, accuracies_val, accuracies_test



def evaluate_embedding(embeddings, labels, classifier, search=True):

    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    acc = 0
    acc_val = 0

    if classifier == 'logistic':
        _acc_train, _acc_val, _acc = logistic_classify(x, y)

    elif classifier == 'neural':
        _acc_train, _acc_val, _acc = nerual_classify(x, y)

    elif classifier == 'svc':
        _acc_train, _acc_val, _acc = svc_classify(x,y, search)
       
    elif classifier == 'linearsvc':
        _acc_train, _acc_val, _acc = linearsvc_classify(x, y, search)
        
    elif classifier == 'randomforest':
        _acc_train, _acc_val, _acc = randomforest_classify(x, y, search)
        
    if np.mean(_acc_val) > acc_val:
            acc_train = np.mean(_acc_train)
            var_train = np.var(_acc_train)
            acc_val = np.mean(_acc_val)
            var_val = np.var(_acc_val)
            acc = np.mean(_acc)
            var = np.var(_acc)
    
    output = 'Train Acc: {} +- {}, Val Acc: {} +- {}, Test Acc: {} +- {}'
    logger.info(output.format(acc_train, var_train, acc_val, var_val, acc, var))

    return acc_train, var_train, acc_val, var_val, acc, var
    

def singular(latents):
    z = torch.from_numpy(latents)
    z = torch.nn.functional.normalize(z, dim=1)

    # calculate covariance
    z = z.cpu().detach().numpy()
    z = np.transpose(z)
    c = np.cov(z)
    _, d, _ = np.linalg.svd(c)

    return np.log(d)


def main():
    #load dataset and get embeddings
    dataset = TUDataset('.data', name='NCI1')
    batch_size = 64
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers = 10, shuffle=True)
    
    model_save_name = 'alpha2.5ginlstmNCI1.pt'
    path = F"./save/{model_save_name}" 
    graph_encoder = torch.load(path, map_location=torch.device('cpu'))
    logger.info(model_save_name)
    
    
    embs, y = graph_encoder.get_embeddings(data_loader)
    '''
    embs = []
    y = []
    with torch.no_grad():
        for data in data_loader:
            for i in range(len(data)):
                x, edge_index, edge_attr= data[i].x, data[i].edge_index, data[i].edge_attr
                if x is None:
                    x = torch.ones((batch.shape[0],1))
                edge_weight = torch.zeros(edge_attr.shape[0], 1)
                edge_attr = torch.cat([edge_weight, edge_attr], dim=1)
                emb = graph_encoder.forward_emb(x, edge_index, edge_attr)
                
                #num_nodes = data[i].num_nodes
                #x = torch.tensor(list(range(num_nodes))).reshape(num_nodes, 1).float()
                #edge_index = data[i].edge_index
                #emb = graph_encoder.forward_emb(x, edge_index)
                embs.append(emb.cpu().numpy())
                y.append(data[i].y.cpu().numpy())

    embs = np.concatenate(embs, 0)
    y = np.concatenate(y, 0)
    '''
    logger.info('Get embeddings')

    #computer spectrum
    spectrum = singular(embs)
    logger.info(spectrum)
    
    evaluate_embedding(embs, y, 'logistic', True)
    evaluate_embedding(embs, y, 'svc', True)
    evaluate_embedding(embs, y, 'linearsvc', True)
    evaluate_embedding(embs, y, 'randomforest', True)

   

if __name__ == "__main__":
    main()









