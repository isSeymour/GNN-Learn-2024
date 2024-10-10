https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/DeepWalk/header.png

# Deep Walk 图嵌入：维基百科词条

## 1. 环境参考

### 参考资料
https://github.com/prateekjoshi565/DeepWalk


### 安装工具包


```python
!pip install networkx gensim pandas numpy tqdm scikit-learn matplotlib
```




### 导入工具包


```python
# 图数据挖掘
import networkx as nx

# 数据分析
import pandas as pd
import numpy as np

# 随机数与进度条
import random
from tqdm import tqdm

# 数据可视化
import matplotlib.pyplot as plt
%matplotlib inline

```

## 2. 数据

### 获取数据

爬虫网站：https://densitydesign.github.io/strumentalia-seealsology/

1. 设置 distance

2. 输入链接：

https://en.wikipedia.org/wiki/Computer_vision

https://en.wikipedia.org/wiki/Deep_learning

https://en.wikipedia.org/wiki/Convoutional_neural_network

https://en.wikipedia.org/wiki/Decision_tree

https://en.wikipedia.org/wiki/Support_vector_machine


3. 点击 `START CRAWLING`, 爬取完成点击 `STOP`

4. Download 下载为 TSV 文件（以`\t`分割的 CSV 文件）。


```python
df = pd.read_csv("seealsology-data.tsv", sep='\t')
```


```python
df.head()
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
      <th>source</th>
      <th>target</th>
      <th>depth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>support vector machine</td>
      <td>in situ adaptive tabulation</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>support vector machine</td>
      <td>kernel machines</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>support vector machine</td>
      <td>fisher kernel</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>support vector machine</td>
      <td>platt scaling</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>support vector machine</td>
      <td>polynomial kernel</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (4232, 3)



### 构建无向图


```python
G = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr=True, create_using=nx.Graph())
```


```python
# 节点个数
len(G)
```




    3059



### 可视化


```python
plt.figure(figsize=(15, 14))
nx.draw(G)
plt.show()
```


​    
![png](https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/DeepWalk/output_16_0.png)
​    


## 3. 随机游走

### randomwalk 函数


```python
def get_randomwalk(node, path_length):
    '''
    输入起始节点和路径长度，生成随机游走节点序列
    '''
    random_walk = [node]
    for i in range(path_length-1):
        # 汇总邻接节点
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break
        # 从邻接节点中随机选择下一个节点
        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node
        
    return random_walk

```


```python
all_nodes = list(G.nodes())
```


```python
get_randomwalk('computer vision', 5)
```




    ['computer vision',
     'machine vision glossary',
     'glossary of artificial intelligence',
     'artificial intelligence',
     'organoid intelligence']



### 生成随机游走序列


```python
gamma = 10   # 每个节点作为起始节点生成随机游走序列个数
walk_length = 5   # 随机游走序列最大长度
```


```python
random_walks = []

for nd in tqdm(all_nodes):   # 遍历每个节点
    for i in range(gamma):   # 每个节点作为起始点生成 gamma 个随机游走序列
        rdwk = get_randomwalk(nd, walk_length)
        random_walks.append(rdwk)

```

    100%|████████████████████████████████████████████████████████████████████████████| 3059/3059 [00:00<00:00, 30093.95it/s]



```python
# 生成随机游走序列的个数
len(random_walks)
```




    30590




```python
random_walks[0]
```




    ['support vector machine', 'relevance vector machine', 'kernel trick']



## 4. 模型

### 训练 Word2Vec 模型


```python
# 自然语言处理
from gensim.models import Word2Vec
```


```python
model = Word2Vec(vector_size=256, # Embedding 维数
                 window=4,        # 窗口宽度
                 sg=1,            # Skip-Gram
                 negative=10,     # 负采样
                 alpha=0.03,      # 初始学习率
                 min_alpha=0.0007,# 最小学习率
                 seed=14          # 随机数种子
                )
```


```python
# 用随机游走序列构建词汇表
model.build_vocab(random_walks, progress_per=2)
```


```python
# 训练（耗时 1 分钟）
model.train(random_walks, total_examples=model.corpus_count, epochs=50, report_delay=1)
```




    (5623725, 5679950)



### 分析 Word2Vec 结果


```python
# 查看某个节点的 Embedding
model.wv.get_vector('computer vision').shape
```




    (256,)




```python
model.wv.get_vector('computer vision')
```




    array([-0.95459837,  0.10292508, -0.28316122,  0.34142157, -0.00524048,
            0.09371996, -0.1954719 , -0.25347382,  0.51394266,  0.36131492,
            0.49506772,  0.1907984 , -0.6219965 , -0.5140934 , -0.01667919,
           -0.62039286, -0.05152594,  0.11786714,  0.18947525,  0.19846195,
           -0.11716247,  0.4700267 ,  0.07052463, -0.17666382,  0.1671837 ,
            0.24031273, -0.18862735, -0.15001939, -0.15928511, -0.13938765,
           -0.05735731, -0.17796549, -0.20125604, -0.13714062,  0.02854507,
           -0.3297002 ,  0.21914023, -0.03728085, -0.42431426,  0.28924662,
           -0.07030115,  0.153452  ,  0.02109604, -0.5424473 , -0.5128256 ,
            0.09319318, -0.18759303, -0.20778346,  0.01962802,  0.2059087 ,
            0.49449265, -0.43316683,  0.47074154,  0.32398415,  0.18804422,
            0.30941215, -0.16319014,  0.5086255 , -0.4054713 ,  0.18189834,
           -0.0757796 ,  0.01394054,  0.29209548, -0.20624508,  0.04370715,
           -0.22285934, -0.1998267 , -0.07965406, -0.56047654,  0.39915815,
           -0.14301345,  0.03823084, -0.51063114, -0.06177189, -0.12064032,
            0.41043568,  0.61430806,  0.00198809, -0.44348234, -0.4718856 ,
            0.17651486,  0.03726299, -0.16133447, -0.07498072,  0.27820274,
            0.4717679 , -0.09105907,  0.23809573,  0.05806234,  0.1386895 ,
           -0.00990544, -0.07417107, -0.13418426,  0.23991434,  0.229925  ,
            0.8267156 ,  0.1580667 ,  0.36089334,  0.09349226,  0.33000064,
            0.191074  ,  0.07245437, -0.19699697,  0.1373127 ,  0.00637828,
           -0.393098  ,  0.08118346, -0.33764714,  0.18177702,  0.6325778 ,
           -0.2885028 , -0.6606645 ,  0.25406113, -0.07453088,  0.0134876 ,
            0.22993505,  0.2469321 , -0.31469256,  0.15289971, -0.2890252 ,
           -0.24749073, -0.60842824, -1.0122712 ,  0.12880209, -0.14758833,
            0.05826454, -0.28706843,  0.14353754, -0.22783504, -0.18525298,
           -0.48144853,  0.03936397, -0.7163454 ,  0.2678299 , -0.03936832,
            0.23881389,  0.47060257, -0.66273224, -0.10196779,  0.5657661 ,
           -0.21970046, -0.11473361,  0.01603065, -0.17330663, -0.07658403,
           -0.00363667,  0.30719343,  0.05218068, -0.0915609 ,  0.18364   ,
           -0.05932966, -0.12060771,  0.29323366, -0.68775976,  0.4539725 ,
            0.3334422 , -0.45317262,  0.3847841 , -0.15240075,  0.11145896,
           -0.5170747 ,  0.28762746,  0.33697945,  0.0671319 ,  0.41540784,
            0.530296  ,  0.7281354 ,  0.3821813 ,  0.05093963,  0.7988582 ,
           -0.38773486, -0.21942078, -0.03484021,  0.3349887 , -0.19996904,
            0.37933737, -0.26954234,  0.4171879 ,  0.77916664, -0.1828221 ,
           -0.19539501, -0.4173407 ,  0.72097695, -0.03344366,  0.07354128,
            0.17265108, -0.4285512 , -0.41779858,  0.31622657,  0.23919132,
           -0.14859721, -0.112137  , -0.62065303,  0.02263851,  0.03000049,
           -0.31004304,  0.16809928,  0.27590737,  0.30516142, -0.2884869 ,
           -0.52874154, -0.0075765 , -0.22995523, -0.5217325 ,  0.61138886,
            0.26653954,  0.11882886,  0.8872766 ,  0.32643762, -0.16740482,
            0.03697263, -0.26058164, -0.5465761 , -0.19003482, -0.14713594,
            0.29176036, -0.15662532, -0.3437838 , -0.6559339 ,  0.29693472,
            0.01657276,  0.10343892, -0.01626491, -0.03184415, -0.15561788,
           -0.39298484, -0.10999571, -0.29130518,  0.49602684,  0.1284142 ,
            0.1823952 , -0.299319  , -0.35532302, -0.31292355,  0.5582348 ,
            0.19172785, -0.29422763,  0.32814986, -0.17529616, -0.3650768 ,
           -0.3434801 , -0.13502142,  0.19740753, -0.15909001, -0.26023048,
            0.22111997,  0.45001796,  0.14510933,  0.40188378,  0.23440124,
            0.02278174, -0.28787047, -0.13803658,  0.12221967, -0.00340613,
            0.03851813], dtype=float32)




```python
# 找相似词语
model.wv.similar_by_word('computer vision')
```




    [('computational imaging', 0.7198930978775024),
     ('teknomo–fernandez algorithm', 0.6524918079376221),
     ('vectorization (image tracing)', 0.6257413625717163),
     ('h-maxima transform', 0.6218506097793579),
     ('egocentric vision', 0.6183371543884277),
     ('multispectral imaging', 0.6168951988220215),
     ('sound recognition', 0.6164405345916748),
     ('ridge detection', 0.6114603281021118),
     ('google goggles', 0.6109517216682434),
     ('medical intelligence and language engineering lab', 0.6081306338310242)]



## 5. PCA 降维可视化

### 全部词条


```python
# 可视化全部词条的二维 Embedding
X = model.wv.vectors
```


```python
from sklearn.decomposition import PCA

# 将 Embedding 用 PCA 降维到 2 维
pca = PCA(n_components=2)
embed_2d = pca.fit_transform(X)
```


```python
embed_2d.shape
```




    (3059, 2)




```python
plt.figure(figsize=(14,14))
plt.scatter(embed_2d[:, 0], embed_2d[:, 1])
plt.show()
```


​    
![png](https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/DeepWalk/output_42_0.png)
​    


### 某个词条


```python
# 可视化某个词条的二维 Embedding
term = 'computer vision'
term_256d = model.wv[term].reshape(1, -1)
term_256d.shape
```




    (1, 256)




```python
term_2d = pca.transform(term_256d)
```


```python
term_2d
```




    array([[-0.6757479,  0.6024744]], dtype=float32)




```python
plt.figure(figsize=(14,14))
plt.scatter(embed_2d[:, 0], embed_2d[:, 1])
plt.scatter(term_2d[:, 0], term_2d[:, 1], c='r', s=200)
plt.show()
```


​    
![png](https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/DeepWalk/output_47_0.png)
​    


### 某些词条


```python
# 可视化某些词条的二维 Embedding
# 计算 PageRank 重要度
pagerank = nx.pagerank(G)
# 从高到低排序
node_importance = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
```


```python
# 取最高的 n 个节点
n = 30
terms_chosen = []
for nd in node_importance[:n]:
    terms_chosen.append(nd[0])

```


```python
# 手动补充新节点
terms_chosen.extend(['computer vision', 'deep learning'])
```


```python
terms_chosen
```




    ['cloud computing',
     'electromagnetic wave equation',
     'spatial dependence',
     '3d modeling',
     'empathy',
     'psychoacoustics',
     'evolutionary psychology',
     'superlens',
     'wearable computer',
     'cognitive science',
     'decision theory',
     'system dynamics',
     'accessibility',
     'brain–computer interface',
     'simulated consciousness',
     'visual perception',
     'artificial neural network',
     'turing test',
     'cognitive psychology',
     'recognition of human individuals',
     'transhumanism',
     'speech repetition',
     'embodied cognition',
     'finite element method',
     'computational neuroscience',
     'fourier analysis',
     'interval finite element',
     'n170',
     'graphical user interface',
     'tensor',
     'computer vision',
     'deep learning']




```python
# 输入词条，输出词典中的索引号
term2index = model.wv.key_to_index
# 反之
index2term = model.wv.index_to_key
```


```python
term_index = np.array(term2index.values())
```


```python
# 可视化全部词条和关键词的二维 Embedding
plt.figure(figsize=(14,14))
plt.scatter(embed_2d[:, 0], embed_2d[:, 1])

for item in terms_chosen:
    idx = term2index[item]
    plt.scatter(embed_2d[idx, 0], embed_2d[idx, 1], c='r', s=50)
    plt.annotate(item, xy=(embed_2d[idx, 0], embed_2d[idx, 1]), c='k', fontsize=12)

plt.show()
```


​    
![png](https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/DeepWalk/output_55_0.png)
​    


## 6. TSNE 降维可视化

### 全部词条


```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, n_iter=1000)
embed_2d = tsne.fit_transform(X)
```

    /opt/anaconda3/envs/graph/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:1162: FutureWarning: 'n_iter' was renamed to 'max_iter' in version 1.5 and will be removed in 1.7.
      warnings.warn(



```python
plt.figure(figsize=(14,14))
plt.scatter(embed_2d[:,0], embed_2d[:,1])
plt.show()
```


​    
![png](https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/DeepWalk/output_59_0.png)
​    


### 某些词条


```python
# 可视化全部词条和关键词的二维 Embedding
plt.figure(figsize=(14,14))
plt.scatter(embed_2d[:, 0], embed_2d[:, 1])

for item in terms_chosen:
    idx = term2index[item]
    plt.scatter(embed_2d[idx, 0], embed_2d[idx, 1], c='r', s=50)
    plt.annotate(item, xy=(embed_2d[idx, 0], embed_2d[idx, 1]), c='k', fontsize=12)

plt.show()
```


​    
![png](https://cdn.jsdelivr.net/gh/isSeymour/PicGo/posts/CS224W/DeepWalk/output_61_0.png)
​    



```python
embed_2d.shape
```




    (3059, 2)



### 导出


```python
terms_chosen_mask = np.zeros(X.shape[0])
for item in terms_chosen:
    idx = term2index[item]
    terms_chosen_mask[idx] = 1

```


```python
df = pd.DataFrame()
df['X'] = embed_2d[:, 0]
df['Y'] = embed_2d[:, 1]
df['item'] = model.wv.index_to_key
df['pagerank'] = pagerank.values()
df['chosen'] = terms_chosen_mask
```


```python
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
      <th>X</th>
      <th>Y</th>
      <th>item</th>
      <th>pagerank</th>
      <th>chosen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-13.242397</td>
      <td>-42.560059</td>
      <td>cloud computing</td>
      <td>0.001352</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42.664997</td>
      <td>13.116780</td>
      <td>evolutionary psychology</td>
      <td>0.000699</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-12.628043</td>
      <td>42.939220</td>
      <td>visual perception</td>
      <td>0.000623</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17.878042</td>
      <td>-14.927996</td>
      <td>cognitive science</td>
      <td>0.000292</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39.976368</td>
      <td>3.763149</td>
      <td>cognitive psychology</td>
      <td>0.000255</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3054</th>
      <td>2.317700</td>
      <td>-71.532204</td>
      <td>browser isolation</td>
      <td>0.000150</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3055</th>
      <td>15.404965</td>
      <td>0.663236</td>
      <td>neural engineering</td>
      <td>0.000150</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3056</th>
      <td>40.051682</td>
      <td>-20.218977</td>
      <td>level of analysis</td>
      <td>0.000150</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3057</th>
      <td>38.300884</td>
      <td>9.790667</td>
      <td>social cognitive and affective neuroscience</td>
      <td>0.000150</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3058</th>
      <td>3.957638</td>
      <td>-22.789871</td>
      <td>problem solving</td>
      <td>0.000150</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3059 rows × 5 columns</p>
</div>




```python
df.to_csv('tsne_vis_2d.csv', index=False)
```

### 三维 TSNE


```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=3, n_iter=1000)
embed_3d = tsne.fit_transform(X)
```

    /opt/anaconda3/envs/graph/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py:1162: FutureWarning: 'n_iter' was renamed to 'max_iter' in version 1.5 and will be removed in 1.7.
      warnings.warn(



```python
df = pd.DataFrame()
df['X'] = embed_3d[:, 0]
df['Y'] = embed_3d[:, 1]
df['Z'] = embed_3d[:, 1]
df['item'] = model.wv.index_to_key
df['pagerank'] = pagerank.values()
df['chosen'] = terms_chosen_mask
```


```python
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
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>item</th>
      <th>pagerank</th>
      <th>chosen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-5.344084</td>
      <td>-13.896581</td>
      <td>-13.896581</td>
      <td>cloud computing</td>
      <td>0.001352</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.959835</td>
      <td>1.637353</td>
      <td>1.637353</td>
      <td>evolutionary psychology</td>
      <td>0.000699</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-9.399863</td>
      <td>5.780833</td>
      <td>5.780833</td>
      <td>visual perception</td>
      <td>0.000623</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.417569</td>
      <td>-12.603775</td>
      <td>-12.603775</td>
      <td>cognitive science</td>
      <td>0.000292</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.512045</td>
      <td>1.471242</td>
      <td>1.471242</td>
      <td>cognitive psychology</td>
      <td>0.000255</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3054</th>
      <td>-7.936583</td>
      <td>-12.397557</td>
      <td>-12.397557</td>
      <td>browser isolation</td>
      <td>0.000150</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3055</th>
      <td>9.576207</td>
      <td>-11.376499</td>
      <td>-11.376499</td>
      <td>neural engineering</td>
      <td>0.000150</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3056</th>
      <td>18.333593</td>
      <td>-2.940028</td>
      <td>-2.940028</td>
      <td>level of analysis</td>
      <td>0.000150</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3057</th>
      <td>7.171093</td>
      <td>3.361520</td>
      <td>3.361520</td>
      <td>social cognitive and affective neuroscience</td>
      <td>0.000150</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3058</th>
      <td>5.557903</td>
      <td>4.377861</td>
      <td>4.377861</td>
      <td>problem solving</td>
      <td>0.000150</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3059 rows × 6 columns</p>
</div>




```python
df.to_csv('tsne_vis_3d.csv', index=False)
```

## 7. 课后作业*

> 用 `tsne_vis_2d.csv` 和 `tsne_vis_3d.csv` 做可视化
> 
> 参考代码：https://echarts.apache.org/examples/zh/editor.html?c=scatter3d&gl=1&theme=dark


```python

```
