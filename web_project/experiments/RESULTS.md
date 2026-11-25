
# 多路召回融合优化实验

## 1. 优化目标

解决原始 Snake Merge 等权融合未能充分利用
各召回通道优势的问题，同时降低在线召回耗时。

## 2. 改进方法

- 实现可配置的加权轮询融合算法。
- 通过消融实验分析各召回通道贡献。
- 采用全局时间切分，重新训练 YouTubeDNN。
- 使用验证集选择融合权重。
- 跳过零权重召回通道。
- 增加历史电影过滤与超额召回机制。

## 3. 数据集划分

| 数据集 | 交互数量 | 有效评估用户 |
|---|---:|---:|
| Train | 800164 | - |
| Validation | 23639 | 520 |
| Test | 80298 | 1002 |

按全局时间进行约 80%/10%/10% 划分。
验证集和测试集仅评估训练期至少有5次交互的用户。

## 4. 最终融合权重

| 召回通道 | 权重 |
|---|---:|
| UserPreference | 4 |
| ItemEmbedding | 0 |
| GlobalTrending | 6 |
| YouTubeDNN | 0 |

## 5. 测试集实验结果

| 策略 | HR@100 | Recall@100 | NDCG@100 |
|---|---:|---:|---:|
| UserPreference | 0.8633 | 0.1813 | 0.1479 |
| ItemEmbedding | 0.7305 | 0.1040 | 0.0961 |
| GlobalTrending | 0.8992 | 0.2184 | 0.2225 |
| YouTubeDNN | 0.6936 | 0.0672 | 0.0672 |
| SnakeMerge | 0.9291 | 0.2149 | 0.1711 |
| WeightedMerge | 0.9411 | 0.2670 | 0.2224 |

相比原始 Snake Merge：

- Recall@100 相对提升 24.24%。
- NDCG@100 相对提升 29.98%。

## 6. 在线召回模块性能测试

| 指标 | SnakeMerge | WeightedMerge |
|---|---:|---:|
| 平均耗时(ms) | 316.35 | 86.45 |
| 中位耗时(ms) | 313.89 | 84.52 |
| 候选数量 | 100 | 100 |
| 重复电影 | 0 | 0 |
| 历史电影重合数 | 0 | 0 |

平均召回耗时降低约72.67%。

## 7. 实验复现顺序

在 web_project 目录下，使用已配置的 Python 环境：

1. temporal_split.py：全局时间切分
2. prepare_temporal_data.py：构建训练样本
3. train_temporal_retrieval.py：训练召回模型
4. eval_temporal_val.py：验证集权重选择
5. eval_temporal_test.py：固定权重最终评估

统一执行方式：

PYTHONPATH=backend:../src backend/.venv/bin/python experiments/<script_name>.py

eval_recall_baseline.py 为早期探索性实验脚本，
不属于最终时间切分评估流程。

## 8. 实验局限性

- 实验仅评估正常用户，不包含冷启动用户。
- 性能测试为单用户预热后10次调用，不代表高并发性能。
- 离线指标不等同于线上业务收益。
- 线上与离线特征处理仍存在一定差异。
- 新测试集可能与早期探索实验存在交互重叠。
- 尚未完成精排和重排的独立效果评估。
