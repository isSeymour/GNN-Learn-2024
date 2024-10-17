# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gnn_task
   Description :
   Author :       Lr
   date：          2023/6/4
-------------------------------------------------
   Change Activity:
                   2023/6/4:
-------------------------------------------------
"""



import dgl
import pandas as pd
import torch
import torch.nn.functional as F
import dgl
import torch
import torch.nn as nn
import itertools



def load_data():
    nodes_data = pd.read_csv('data/nodes.csv')
    edges_data = pd.read_csv('data/edges.csv')
    src = edges_data['Src'].to_numpy()
    dst = edges_data['Dst'].to_numpy()
    g = dgl.graph((src, dst))
    club = nodes_data['Club'].to_list()

    club = torch.tensor([c == 'Club1' for c in club]).long()

    club_onehot = F.one_hot(club)
    g.ndata.update({'club' : club, 'club_onehot' : club_onehot})
    return g



g = load_data()
# print(g)



# ----------- 1. node features -------------- #
node_embed = nn.Embedding(g.number_of_nodes(), 5)
inputs = node_embed.weight
nn.init.xavier_uniform_(inputs)
print(inputs)
print(g.ndata['club'])

labels = g.ndata['club']
labeled_nodes = [1, 22]

print('Labels', labels[labeled_nodes])

from dgl.nn import SAGEConv


# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, num_classes, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h



net = GraphSAGE(5, 16, 2)


optimizer = torch.optim.Adam(itertools.chain(net.parameters(), node_embed.parameters()), lr=0.01)

all_logits = []
print(labeled_nodes)
for e in range(100):
    logits = net(g, inputs)

    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[labeled_nodes], labels[labeled_nodes])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    all_logits.append(logits.detach())

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))



pred = torch.argmax(logits, axis=1)
print('Accuracy', (pred == labels).sum().item() / len(pred))