# FunRec 个人 PyTorch 推荐算法学习与复现实验

本目录是基于 [Datawhale FunRec](https://github.com/datawhalechina/fun-rec) 教程开展的**个人练习代码**，与仓库中的上游 `src/funrec/` 官方实现区分维护。目标是独立理解模型关键组件、手写 PyTorch 训练/推理代码、比较离线结果；**不是完整工业推荐系统**，也不代表所有论文模型的逐项完整复现。

## 个人练习结构

| 方向 | 文件位置 | 实践内容与当前范围 |
| --- | --- | --- |
| I2I 召回 | `I2I/word2vec/`, `I2I/item2vec/` | Skip-gram 正样本构造、Gensim Item2Vec、MovieLens-1M 用户向量召回与基线评估 |
| I2I 延伸 | `I2I/eges/`, `I2I/airbnb/` | EGES 加权嵌入、Airbnb 上下文相关模块练习；不声明完成全流程离线评估 |
| 双塔/特征交互 | `two_tower/` | FM 二阶交互、DSSM 用户/物品编码、YouTubeDNN 历史池化与采样 Softmax 等模块级 PyTorch 练习 |
| 序列召回 | `Sequential_Retrieval/mind/` | MIND 胶囊网络、多兴趣动态路由、标签感知注意力、训练及验证 |
| 序列召回 | `Sequential_Retrieval/sdm/` | SDM 长短期兴趣、LSTM/多头注意力、门控融合、训练与 Top-10 验证 |
| 流式索引 | `Streaming_Index/trinity/` | Trinity 聚类相关学习代码；仍在练习，不作为已完成系统 |

序列召回的详细评估协议与限制见 [Sequential_Retrieval/README.md](Sequential_Retrieval/README.md)。

## 数据与已核验实验

### Item2Vec / MovieLens-1M（2026-10-28）

1. 仅保留 `rating >= 4` 作为正反馈，按用户和时间排序，并保留至少两条正反馈的用户。
2. 每个用户末次正反馈作为测试目标，其余历史训练 Item2Vec；使用平均历史电影向量生成用户表征。
3. 过滤已看电影，计算 HR@10，并与同数据划分下的 Popularity 比较。

| 方法 | HR@10 |
| --- | ---: |
| Item2Vec（历史向量平均池化） | 1.5239% |
| Popularity | 4.0749% |

测试用户 6,037 人，Item2Vec 命中 92 人；Warm HR@10 约为 1.5247%，Cold Item Rate 约为 0.0497%。**本次实验中热门推荐更好**；仅测试过单组参数，不代表 Item2Vec 在其他条件下普遍较差。训练词表外的电影不可被该 Item2Vec 模型召回；历史过滤也可能排除用户重复交互的测试目标。

### MIND / SDM（MovieLens-1M，验证集）

- 正反馈采用 `rating >= 4`；按用户时间序列构造训练前缀、验证目标与留出测试目标。
- 当前报告的是**验证集** HR@10，尚未使用独立测试集完成最终对比。
- SDM 为简化实现：最近最多 10 条历史构造短期兴趣，之前最多 40 条构造长期兴趣；未完整复现论文的多特征建模。

| 方法 / 配置 | Val HR@10 |
| --- | ---: |
| Popularity | 3.76% |
| MIND（routing_std=0.5） | 5.00% |
| SDM（1 Epoch） | 4.6396% |
| SDM（3 Epoch） | 6.8600% |
| SDM（5 Epoch） | 8.8981% |

SDM（5 Epoch）命中 537/6035，Warm HR@10 约为 8.9055%。不同模型训练预算、输入长度及参数配置并未完全对齐，**不能将此表解释为公平的架构优劣比较**。详情、现有检查点和运行说明见序列召回 README。

## 使用与复现说明

- 推荐使用 Python 3.12、PyTorch、NumPy、Pandas、Gensim；具体依赖视练习脚本而定。
- 练习按独立脚本组织，**没有统一的一键训练入口**；部分脚本仅演示张量维度、前向计算或局部训练步骤。
- MovieLens-1M 的 `ratings.dat` 需由使用者自行获取；部分练习中的本地路径尚需手动修改。
- 序列召回评估脚本依赖先行训练并保存的本地模型检查点；不上传 `.pt` 权重。
- Item2Vec 模型文件 `.model` 也不上传；换机器运行评测之前需按同样切分与参数重新训练。
- 训练集、验证集、测试集和线上业务指标不能混同；这里没有线上 A/B 测试、真实服务或全流程性能保障。

## 代码来源与边界

本仓库为 [Datawhale FunRec](https://github.com/datawhalechina/fun-rec) 的个人 Fork。原有 `src/funrec/` 和教程文档属于上游项目；`practice/` 下的练习是本人跟随教程与源码独立编写/调试的实现。部分模型只是关键组件重写，不应写作“原创提出模型”或“完整复现官方所有模块”。

**求职简历建议**：优先强调 PyTorch Embedding/Attention、数据集与 Mask、负采样、训练循环、Top-K 评估，以及 MIND/SDM 的实验差异与边界；不要将上述验证指标包装为生产环境收益。
