import dgl
import dgl.function as fn
import torch


# 创建一个简单的DGL图
g = dgl.graph(([0, 1, 2], [1, 2, 3]))

# 添加节点特征
g.ndata['h'] = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

# 使用u_dot_v计算点积
dot_product = fn.u_dot_v('h', 'h',"score")

# 应用u_dot_v函数，计算每个边的特征点积
g.apply_edges(dot_product)

print(g)
print(g.edata["score"])

# # 获取边的特征点积
# edge_dot_product = g.edata['dot_product']
#
# print(edge_dot_product)

import math

print(math.floor(15/4))
print(math.ceil(15/4))
print(math.ceil(15/5))