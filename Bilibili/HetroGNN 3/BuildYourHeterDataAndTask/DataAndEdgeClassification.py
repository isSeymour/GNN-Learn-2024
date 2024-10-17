# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     DataAndEdgeClassification
   Description :
   Author :       Lr
   date：          2023/9/1
-------------------------------------------------
   Change Activity:
                   2023/9/1:
-------------------------------------------------
"""



import dgl
import dgl.nn as dglnn
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F


n_users = 2000 #用户数
n_video = 500 #视频数
n_follows = 3000 # 用户互相关注 的关系 数
n_clicks = 5000 # 用户点击数
n_dislikes = 500 # 用户不喜欢数
n_hetero_features = 10 # 节点特征维度
n_user_classes = 5 # 用户类型数

# 边回归 额外添加
n_max_clicks = 10 #


#np.random.randint(low, high, size, dtype='l')
#采用边-3000
follow_src = np.random.randint(0, n_users, n_follows)
follow_dst = np.random.randint(0, n_users, n_follows)
#点击边-5000
click_src = np.random.randint(0, n_users, n_clicks)
click_dst = np.random.randint(0, n_video, n_clicks)
#不喜欢边-500
dislike_src = np.random.randint(0, n_users, n_dislikes)
dislike_dst = np.random.randint(0, n_video, n_dislikes)

hetero_graph = dgl.heterograph({  # 正反两个方向构边
    ('用户', '关注', '用户'): (follow_src, follow_dst),
    ('用户', '被关注', '用户'): (follow_dst, follow_src),

    ('用户', '点击', '视频'): (click_src, click_dst),
    ('视频', '被点击', '用户'): (click_dst, click_src),

    ('用户', '喜欢', '视频'): (dislike_src, dislike_dst),
    ('视频', '被喜欢', '用户'): (dislike_dst, dislike_src)
})


print(hetero_graph)



#特征构造

#用户特征，2000个用户特征10维
hetero_graph.nodes['用户'].data['feature'] = torch.randn(n_users, n_hetero_features)
#项目特征，500个视频特征10维
hetero_graph.nodes['视频'].data['feature'] = torch.randn(n_video, n_hetero_features)
#用户类型标签,2000维向量
hetero_graph.nodes['用户'].data['label'] = torch.randint(0, n_user_classes, (n_users,))



#边回归 额外添加
hetero_graph.edges['点击'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()
hetero_graph.nodes['用户'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
# 边回归  额外添加
hetero_graph.edges['点击'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)



import dgl.function as fn
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):

        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']




# 定义特征聚合模块
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feats, hid_feats)
                                            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hid_feats, out_feats)
                                            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype)


import torchmetrics


model = Model(10, 20, 5, hetero_graph.etypes)
user_feats = hetero_graph.nodes['用户'].data['feature']
item_feats = hetero_graph.nodes['视频'].data['feature']
label = hetero_graph.edges['点击'].data['label']
train_mask = hetero_graph.edges['点击'].data['train_mask']
node_features = {'用户': user_feats, '视频': item_feats}


opt = torch.optim.Adam(model.parameters())
for epoch in range(700):
    pred = model(hetero_graph, node_features, '点击')
    loss = ((pred[train_mask] - label[train_mask]) ** 2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    rmse = torchmetrics.functional.mean_squared_error(pred[train_mask].squeeze(1),label[train_mask])

    print("loss = %.4f , train rmse = %.4f" % (loss.item(),rmse))



