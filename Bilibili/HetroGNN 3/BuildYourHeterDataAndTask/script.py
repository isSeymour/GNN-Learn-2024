# import dgl
# import dgl.function as fn
# import torch


# # 创建一个简单的DGL图
# graph = dgl.graph(([0, 1, 2], [1, 2, 3]))
# k = 2

# src, dst = graph.edges()
# print(f"src type = {type(src)}")
# neg_src = src.repeat_interleave(k)

# neg_dst = torch.randint(0, graph.number_of_nodes(), (len(src) * k,))

# print(src)
# print(f" neg_src = {neg_src}")
# print(f"neg_dst = {neg_dst}")



r,c = map(int,[2,3])
print(r)
print(c)
