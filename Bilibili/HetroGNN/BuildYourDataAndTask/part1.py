# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     load_data
   Description :
   Author :       Lr
   date：          2023/6/4
-------------------------------------------------
   Change Activity:
                   2023/6/4:
-------------------------------------------------
"""



import pandas as pd

nodes_data = pd.read_csv('data/nodes.csv')
# print(nodes_data)

edges_data = pd.read_csv('data/edges.csv')
# print(edges_data)



import dgl

src = edges_data['Src'].to_numpy()
dst = edges_data['Dst'].to_numpy()

g = dgl.graph((src, dst))

# print(g)


print('#Nodes', g.number_of_nodes())
print('#Edges', g.number_of_edges())

import torch
import torch.nn.functional as F

club = nodes_data['Club'].to_list()
club = torch.tensor([c == 'Officer' for c in club]).long()
# We can also convert it to one-hot encoding.
club_onehot = F.one_hot(club)
print(club_onehot)

g.ndata.update({'club' : club, 'club_onehot' : club_onehot})

print(g)