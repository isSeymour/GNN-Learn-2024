# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     DataAndNodeClassification
   Description :
   Author :       Lr
   date：          2023/7/22
-------------------------------------------------
   Change Activity:
                   2023/7/22:
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

# n_max_clicks = 10 #


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



#边标签
# hetero_graph.edges['点击'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()

hetero_graph.nodes['用户'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
# hetero_graph.edges['点击'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)


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



#构造模型
model = RGCN(in_feats=n_hetero_features, hid_feats=20,
             out_feats=n_user_classes, rel_names=hetero_graph.etypes)

user_feats = hetero_graph.nodes['用户'].data['feature'] #用户特征
item_feats = hetero_graph.nodes['视频'].data['feature'] #项目特征
labels = hetero_graph.nodes['用户'].data['label'] #用户类型标签
train_mask = hetero_graph.nodes['用户'].data['train_mask']



node_features = {'用户':user_feats, '视频':item_feats}
#特征字典
h_dict = model(hetero_graph, node_features)

# 模型优化器
opt = torch.optim.Adam(model.parameters())

best_train_acc = 0
loss_list = []
train_score_list = []

# 迭代训练
for epoch in range(500):
    model.train()
    # 输入图和节点特征，提取出user的特征
    logits = model(hetero_graph, node_features)['用户']
    # 计算损失
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    # 预测user
    pred = logits.argmax(1)
    # 计算准确率
    train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
    if best_train_acc < train_acc:
        best_train_acc = train_acc
    train_score_list.append(train_acc)

    # 反向优化
    opt.zero_grad()
    loss.backward()
    opt.step()
    loss_list.append(loss.item())
    #输出训练结果
    print('Loss %.4f, Train Acc %.4f (Best %.4f)' % (
        loss.item(),
        train_acc.item(),
        best_train_acc.item(),))