# FunRec 个人学习实践

本目录用于在阅读 [Datawhale FunRec](https://github.com/datawhalechina/fun-rec) 教程的过程中，独立实现核心练习、理解源码并验证推荐效果。此目录为个人 Fork 新增内容，与上游官方实现分开维护。

## 目录结构

```text
practice/
├── word2vec/
│   └── skipgram_pairs.py       # Skip-gram 滑动窗口与正样本构造
└── item2vec/
    ├── item2vec_train.py       # 小规模商品序列：向量训练与 I2I 相似度
    └── item2vec_movielens.py   # MovieLens-1M：U2I 召回、冷启动统计、基线对比
```

## 实践内容

### 1. Skip-gram 训练样本

`word2vec/skipgram_pairs.py` 使用滑动窗口，为每个中心词生成上下文正样本；主要练习 `enumerate()`、`range()`、列表与索引边界。**不包含完整的 Word2Vec 神经网络训练**。

### 2. Item2Vec：玩具商品数据

`item2vec/item2vec_train.py` 基于 Gensim Word2Vec（`sg=1`）学习商品向量，通过 NumPy 实现余弦相似度计算、已查询商品过滤、Top-3 排序。仅用于验证数据结构与算法流程，样本量很小，不用于评价实际推荐质量。

### 3. Item2Vec：MovieLens-1M

`item2vec/item2vec_movielens.py` 读取 MovieLens-1M 评分数据，按以下流程构造并评测推荐系统：

1. 只保留 `rating >= 4` 的正反馈。
2. 按用户 ID、时间戳排序；保留至少有 2 条正反馈的用户。
3. 每位用户最后一条正反馈作测试目标，之前的交互作训练历史（Leave-Last-Out）。
4. 使用训练历史学习电影 Embedding；以历史电影向量的**均值**作为用户向量。
5. 依据向量相似度召回，过滤用户已交互电影，计算 `HR@10` 与 `Warm HR@10`。
6. 在相同数据切分下，计算 Popularity Top-10 基线。

**Word2Vec 训练配置：**

| 参数 | 值 |
| --- | --- |
| `vector_size` | 32 |
| `window` | 5 |
| `min_count` | 1 |
| `sg` | 1（Skip-gram） |
| `workers` | 1 |
| `epochs` | 10 |
| `seed` | 42 |

## 实验结果

实验日期：2026-10-28。数据：MovieLens-1M，正反馈评分阈值 4，Leave-Last-Out，排除历史已交互电影。

| 模型 | HR@10 |
| --- | ---: |
| Item2Vec（平均池化用户向量） | **1.5239%** |
| Popularity | **4.0749%** |

辅助指标：Item2Vec 的 `Warm HR@10 = 1.5247%`，`Cold Item Rate = 0.0497%`。测试用户 **6,037** 人，其中测试电影属于训练词表的用户 **6,034** 人、冷启动测试电影对应用户 **3** 人；Item2Vec 命中 **92** 人。

**结果解读：** 该实验中 Popularity 优于 Item2Vec；目前只测试了单组超参数，且用户表示使用均值池化，不能由此推断 Item2Vec 一般不如热门推荐。当前向量召回无法直接召回训练词表外的冷启动电影。

**评测边界：** 由于按交互记录留出最后一次正反馈，若某部电影之前已出现在同一用户的训练历史中，则测试目标会被历史过滤规则排除，成为不可命中样本。该实验尚未对这种重复交互情形单独做调整。离线指标仅供本次练习使用，不可直接与其他数据划分上的指标比较。

## 运行说明

在仓库根目录执行（示例环境为 WSL / Linux）：

```bash
# 本地 Gensim、NumPy、Pandas 环境准备完毕后
python practice/word2vec/skipgram_pairs.py
python practice/item2vec/item2vec_train.py

# MovieLens 评测：先配置数据路径并准备已训练模型
python practice/item2vec/item2vec_movielens.py
```

依赖包：`gensim`、`numpy`、`pandas`。具体可按你的 Python 环境安装；仓库根目录已有独立的官方依赖配置。

**数据集：** [MovieLens-1M（GroupLens）](https://grouplens.org/datasets/movielens/1m/)。下载并解压后，把 `item2vec_movielens.py` 中的 `path` 改为本机 `ratings.dat` 的路径（格式为 `user_id::movie_id::rating::timestamp`）。

**模型文件与复现注意：** 当前 `item2vec_movielens.py` 使用 `Word2Vec.load("practice/item2vec/item2vec_ml1m.model")`，因此新环境运行评测前必须先用上述训练配置、相同数据切分生成并保存该模型；仅克隆 GitHub 仓库不能直接运行这份评测脚本。模型产物通过 `.gitignore` 排除，不上传 GitHub。运行命令应在仓库根目录执行，确保相对模型路径正确。

一个最小的模型生成片段（在数据预处理产生 `train_histories` 后执行）：

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=train_histories,
    vector_size=32,
    window=5,
    min_count=1,
    sg=1,
    workers=1,
    epochs=10,
    seed=42,
)
model.save("practice/item2vec/item2vec_ml1m.model")
```

上述模型训练与当前评测脚本分开运行，避免每次调试评测都重复训练。

## 学习记录

- 熟悉 Python 集合、列表推导式、`zip()`、`Counter`，以及 Pandas `groupby().agg(list)` 等数据处理操作。
- 理解 Skip-gram 样本构造与商品 Embedding 的训练、查询。
- 实现基于用户历史平均 Embedding 的 Top-K 召回，完成冷启动统计及 Popularity 基线对比。
- 后续按 FunRec 教程继续学习 EGES、DSSM、YouTubeDNN、精排等模型。

> 本目录是学习练习，不应被描述为生产级推荐系统或已经完成的工业部署项目。
