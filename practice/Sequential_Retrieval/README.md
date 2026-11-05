# 2.4 序列召回实践：MIND / SDM

本目录是学习 [Datawhale FunRec](https://github.com/datawhalechina/fun-rec) 序列召回章节时的个人 PyTorch 练习；不等同于完整复现官方模型，也不是生产环境实现。

## 文件结构

- `mind/`：MIND 多兴趣召回，包含胶囊网络、动态路由、训练与评估脚本。
- `sdm/`：SDM 长短期兴趣建模、训练与评估脚本。
- 各模型的 `*.pt` 权重为本地产物，不提交 GitHub。

## 数据及划分

- 数据集：MovieLens-1M 的 `ratings.dat`；只保留 `rating >= 4`。
- 用户交互按时间排序；不足 4 条正反馈的用户不参与实验。
- 每个用户：倒数第二条作验证目标，最后一条留作测试目标，其余前缀构造滑动窗口训练样本。
- 电影词表仅由训练前缀构建；验证目标不在词表中的用户计入总体 HR@10 分母，但无法被模型命中。
- **目前仅报告验证集结果，未进行独立测试集最终评估。**

数据路径目前硬编码在 `sdm/sdm_dataset.py` 和 `mind/mind_dataset.py` 中，换机器需改成实际的 `ratings.dat` 路径。

## SDM 简化实现

- 短期兴趣：最近 10 条交互 → Item Embedding → LSTM → Multi-head Attention → User Attention。
- 长期兴趣：此前最多 40 条交互 → Item Embedding → User Attention。
- 门控融合：`sigmoid(W1·user + W2·short + W3·long)`，融合长短期兴趣。
- 历史与候选电影共用 Item Embedding；用正样本 + 4 个采样负样本计算交叉熵。
- 当前实现仅使用 Movie ID 表示长期行为，不包含官方 SDM 的完整多特征建模。

训练配置：Embedding 维度 16、Attention 头数 2、Batch Size 32、Adam 学习率 0.001、固定随机种子 42。

## 验证集结果

| 模型 | 训练设置 | Val HR@10 |
| --- | --- | ---: |
| Popularity | 序列召回练习基线 | 3.76% |
| MIND | routing_std=0.5 | 5.00% |
| SDM | 1 Epoch | 4.6396% |
| SDM | 3 Epoch | 6.8600% |
| SDM | 5 Epoch | **8.8981%** |

SDM 5 Epoch：Warm HR@10 = **8.9055%**；命中 **537/6035**，验证目标冷物品用户 **5**。

说明：
- SDM 训练 5 Epoch 的检查点为 `sdm/sdm_model5.pt`；1 Epoch 为 `sdm/sdm_model.pt`。3 Epoch 结果曾验证，但当前目录没有对应权重文件。
- 上表仅是不同实验配置下的观察值，不构成对 MIND 和 SDM 架构优劣的严格比较。尤其 SDM 只使用最近 50 条输入历史，而 MIND 的输入历史长度和网络、训练参数不同。
- 当前 Top-10 排除 Padding 和传入模型的历史电影；未提供完整的多随机种子、等参数量或统一训练预算对照。
- 未保存每个 Epoch 的结构化 Loss 日志；不能据此判断完整收敛过程。

## 运行方式

在 WSL / Linux 中执行（需预先安装 PyTorch、Pandas，准备 MovieLens-1M 数据）：

```bash
cd /home/dev/projects/fun-rec/practice/Sequential_Retrieval/sdm
/home/dev/projects/fun-rec/.venv/bin/python sdm_train.py
/home/dev/projects/fun-rec/.venv/bin/python sdm_eval.py
```

当前训练脚本设置为 **5 Epoch**，保存 `sdm_model5.pt`；评估脚本默认加载该模型。重新运行训练会覆盖同名权重；更改 Epoch 后应同时检查训练和评估的模型路径。

MIND 练习在同级 `mind/` 目录下；其评估依赖对应的本地检查点。
