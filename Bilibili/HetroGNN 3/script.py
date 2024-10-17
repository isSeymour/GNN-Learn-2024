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

print(str(1234)[:4//2])

ans = 0

for item in range(1200, 1230 + 1):
    itemNum = len(str(item))
    print("itemNum = ", itemNum)

    if itemNum % 2 == 0:
        print("item = ", item)
        print([int(x) for x in str(item)[:itemNum // 2]])
        print([int(x) for x in str(item)[itemNum // 2:]])

        if sum([int(x) for x in str(item)[:itemNum // 2]]) == sum([int(x) for x in str(item)[itemNum // 2:]]):
            ans += 1

print(ans)


a = [1,2,3]
print(a[0:0])

print(int("2908305") % 25)

print(5 // 2)
print(5 % 2)

print(25 %2)
print(13 % 2)

print(-17 % 10)


# 25 % 2  = 1
# 25 - 1 % 2 = 0

print((25-1)%2)

print(" ======== ")
from itertools import accumulate


a = [1,2,3,4,5]
for x in accumulate(a,initial=0):
    print(x)

print(-7 % 2)

print(set([1,2,3]) & set([1,2]))

print(-5%5)



import bisect

# # 一个包含字符串的列表
# word_list = ["apple", "banana", "cherry", "date", "grape", "kiwi"]
#
# # 使用 key 参数来指定一个函数，将字符串长度作为排序依据
# insert_position = bisect.bisect_left(word_list, "cranberry", key=lambda x: len(x))
#
# print(f"插入位置为: {insert_position}")


print(bisect.bisect_left(range(9), True, key=lambda i: i*i>n) - 1)