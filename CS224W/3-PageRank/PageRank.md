# PageRank : 《哈利·波特》人物节点重要度

> 数据来源：http://data.openkg.cn/dataset/a-harry-potter-kg

## 0. 理论知识

### G 矩阵

<img src="https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/PageRank/GoogleMatrix.png" alt="GoogleMatrix" style="zoom:33%;" /> 

> 其中：
>
> - 每个节点A的重要度由指向A 的节点B、C、D重要度决定（B/C/D 本身可能指向了多个，则均摊出去）
> - 为了防止陷入问题，添加 Teleport 传送概率

### 示例

<img src="https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/PageRank/Teleports.png" alt="Teleports" style="zoom:33%;" /> 

<img src="https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/PageRank/Example.png" alt="Example" style="zoom:33%;" /> 

### 可用于推荐系统

<img src="https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/PageRank/Algorithm.png" alt="Algorithm" style="zoom:33%;" /> 

### 总结

> 很巧妙的是，实际上后续的两个变种，就是把传送矩阵的均等传送的概率改成非均等、有偏好的传送，就形成了推荐系统的效果。下面是总结：

<img src="https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/PageRank/Summary.png" alt="Summary" style="zoom:33%;" /> 










## 1. 环境准备


```python
!pip install networkx numpy matplotlib pandas
```


```python
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
# 使用macOS系统自带的中文字体
plt.rcParams['font.family'] = ['STHeiti']
# 设置负号正常显示
plt.rcParams['axes.unicode_minus'] = False
```

## 2. 数据


```python
df = pd.read_csv('harrypotter.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>head</th>
      <th>tail</th>
      <th>relation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C.沃林顿</td>
      <td>斯莱特林魁地奇球队</td>
      <td>从属</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C.沃林顿</td>
      <td>调查行动组</td>
      <td>从属</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C.沃林顿</td>
      <td>霍格沃茨魔法学校</td>
      <td>从属</td>
    </tr>
    <tr>
      <th>3</th>
      <td>乔治·韦斯莱</td>
      <td>亚瑟·韦斯莱</td>
      <td>父亲</td>
    </tr>
    <tr>
      <th>4</th>
      <td>乔治·韦斯莱</td>
      <td>凤凰社</td>
      <td>从属</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1733</th>
      <td>鲍曼·赖特的父亲</td>
      <td>赖特夫人</td>
      <td>妻子</td>
    </tr>
    <tr>
      <th>1734</th>
      <td>鲍曼·赖特的父亲</td>
      <td>鲍曼·赖特</td>
      <td>儿子</td>
    </tr>
    <tr>
      <th>1735</th>
      <td>齐格蒙特·巴奇</td>
      <td>作家</td>
      <td>职业</td>
    </tr>
    <tr>
      <th>1736</th>
      <td>齐格蒙特·巴奇</td>
      <td>巴奇夫人</td>
      <td>母亲</td>
    </tr>
    <tr>
      <th>1737</th>
      <td>齐格蒙特·巴奇</td>
      <td>药剂师</td>
      <td>职业</td>
    </tr>
  </tbody>
</table>
<p>1738 rows × 3 columns</p>
</div>




```python
edges = [edge for edge in zip(df['head'], df['tail'])]

G = nx.DiGraph()
G.add_edges_from(edges)
print(G)
G.nodes
```

    DiGraph with 648 nodes and 1716 edges





    NodeView(('C.沃林顿', '斯莱特林魁地奇球队', '调查行动组', '霍格沃茨魔法学校', '乔治·韦斯莱', '亚瑟·韦斯莱', '凤凰社', '吉迪翁·普威特', '哈利·波特', '塞德瑞拉·布莱克', '塞德里克·迪戈里', '塞普蒂默斯·韦斯莱', '奥黛丽·韦斯莱', '安吉利娜·约翰逊', '弗雷德·韦斯莱', '弗雷德·韦斯莱二世', '普威特先生', '普威特夫人', '普威特家族', '查理·韦斯莱', '格兰芬多学院', '格兰芬多魁地奇球队', '比利尔斯', '比尔·韦斯莱', '珀西·韦斯莱', '纳威·隆巴顿', '罗克珊·韦斯莱', '罗恩·韦斯莱', '芙蓉·德拉库尔', '莫丽·韦斯莱', 
    ...
    '弗林特家族', '马法尔达·霍普柯克', '马琳·麦金农', '麦金农家庭', '格洛普的父亲', '海格先生', '海格家庭', '霍格沃茨保护神奇动物课', '霍格沃茨钥匙保管员和猎场看守', '傲罗办公室主任(曾经)', '魔法部部长(曾经)', '鲁弗斯·福吉', '福吉夫人', '福吉家庭', '鲍曼·赖特的父亲', '赖特夫人', '鲍曼·赖特', '齐格蒙特·巴奇', '巴奇夫人', '药剂师'))




```python
pos = nx.spring_layout(G, iterations=3, seed=5)
nx.draw(G, pos, with_labels=True, font_family='STHeiti')
plt.show()
```


​    
![png](https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/PageRank/PageRank_8_0.png)
​    


## 3. PageRank

### 计算


```python
pagerank = nx.pagerank(G,           # 有向图，无向图则自动转为双向
                       alpha=0.85,  # Damping factor 阻尼系数
                       personalization=None,   # 是否开启，传送到某些节点的概率更高、更低
                       max_iter=100, # 最大迭代次数
                       tol=1e-6,     # 判定收敛的误差
                       nstart=None,  # 每个节点的初始 PageRank
                       dangling=None,# Dead End 死胡同节点
                       )
pagerank
```




    {'C.沃林顿': 0.0011475314330162943,
     '斯莱特林魁地奇球队': 0.0027238782181771616,
     '调查行动组': 0.0018337860389585634,
     '霍格沃茨魔法学校': 0.025270377379097698,
     '乔治·韦斯莱': 0.0016518470047273744,
     '亚瑟·韦斯莱': 0.001919820640437418,
     '凤凰社': 0.0041597878081630684,
     '吉迪翁·普威特': 0.0015170376893856247,
     '哈利·波特': 0.0034783399159879504,
     '塞德瑞拉·布莱克': 0.0015202090673998125,
     '塞德里克·迪戈里': 0.0016668229203842615,
     ...
     '齐格蒙特·巴奇': 0.0011475314330162943,
     '巴奇夫人': 0.0014727490521785147,
     '药剂师': 0.0014727490521785147}




```python
# 从高到低排序
sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
```




    [('霍格沃茨魔法学校', 0.025270377379097698),
     ('美国魔法国会', 0.0111635122688707),
     ('英国魔法部', 0.01055747811015212),
     ('魔法部', 0.00598102537466373),
     ('莫迪丝蒂·巴瑞波恩', 0.00446446534557678),
     ('秋·张', 0.004367509036089586),
     ('凤凰社', 0.0041597878081630684),
     ('小亨利·肖', 0.00396574661310703),
     ('鼻涕虫俱乐部', 0.003716185458476508),
     ('食死徒', 0.0036920571771065606),
     ('斯莱特林', 0.0036914791301060497),
     ('哈利·波特', 0.0034783399159879504),
     ('兰登·肖', 0.0033818974877009474),
     ('老亨利·肖', 0.0033818974877009474),
     ('威廉一世', 0.0033412088969898153),
     ('邓布利多军', 0.003262377574330015),
     ('霍格沃茨', 0.003157988749458249),
     ('梅森家庭', 0.0030988371479896176),
     ('雅各布·科瓦尔斯基', 0.0030988371479896176),
     ('格兰芬多', 0.0030933172566557504),
     ...
     ('鲍曼·赖特的父亲', 0.0011475314330162943),
     ('齐格蒙特·巴奇', 0.0011475314330162943)]



### 尺寸可视化


```python
# 节点尺寸
node_sizes = (np.array(list(pagerank.values())) * 8000).astype(int)
node_sizes
```




    array([  9,  21,  14, 202,  13,  15,  33,  12,  27,  12,  13,  11,  12,
            12,  16,  11,  13,  13,  14,  12,  12,  20,  13,  12,  13,  23,
            12,  16,  16,  14,  12,  12,  24,  26,  15,  11,  13,  10,   9,
            16,  26,  11,   9,  17,  89,  19,   9,   9,  11,   9,   9,  11,
            ...
            10,  10,   9,   9,  11,   9,   9,  13,  10,  10,  10,  10,  10,
            10,  10,   9,  11,  11,   9,  13,  13,   9,  11,  11])




```python
# 颜色
M = G.number_of_edges()
edge_colors = range(2, M+2)

# 绘制节点
nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_sizes)
# 绘制边
edges = nx.draw_networkx_edges(
    G,
    pos,
    node_size=node_sizes,
    arrowstyle='->',
    arrowsize=20,
    edge_color=edge_colors,
    edge_cmap=plt.cm.plasma,   # 连线配色方案，可选plt.cm.Blues
    width=4
)
# 透明度
edge_alphas = [(5+i)/(M+4) for i in range(M)]
for i in range(M):
    edges[i].set_alpha(edge_alphas[i])

ax = plt.gca()
ax.set_axis_off()
plt.show()
```


​    
![png](https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/PageRank/PageRank_14_0.png)
​    



```python

```
