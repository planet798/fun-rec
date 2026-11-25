# FunRec Recsys —— 电影推荐系统

基于 Datawhale Fun-Rec 第 10 章构建的电影推荐系统，基于 MovieLens-1M 数据集，实现从离线模型训练、在线多路召回、精排到前端展示的完整推荐链路。

在官方项目复现的基础上，进一步开展源码分析、召回策略扩展、融合算法优化、时间切分评估及在线推理性能优化。

**上游项目：** [Datawhale Fun-Rec](https://github.com/datawhalechina/fun-rec)

## 1. 项目介绍

本项目采用经典的推荐系统级联架构：

- **召回（Recall）：** 从海量电影中快速筛选用户可能感兴趣的候选电影。
- **精排（Ranking）：** 使用 DeepFM 预测用户对候选电影的偏好。
- **重排（Re-ranking）：** 根据电影类型和年代调整推荐列表，提高结果多样性。

项目支持用户注册、登录、电影搜索、电影详情、评分及个性化推荐。

## 2. 技术栈

| 模块 | 技术 |
|---|---|
| 前端 | Vue 3、Tailwind CSS、Pinia、Vue Router |
| 后端 | Python 3.11、FastAPI |
| 推荐模型 | YouTubeDNN、DeepFM |
| 数据库 | PostgreSQL |
| 缓存 | Redis |
| 搜索 | Elasticsearch |
| 部署 | Docker、Docker Compose、Nginx |
| 模型框架 | TensorFlow 2.15 |

## 3. 系统架构

### 3.1 整体架构

```text
             Vue 3 前端
                  |
                  v
             Nginx 反向代理
                  |
                  v
             FastAPI 服务
                  |
          用户特征与历史行为
                  |
                  v
             冷启动判断
                  |
                  v
              多路召回
          /       |       \
         /        |        \
  UserPreference  ItemEmbedding
         \        |        /
          GlobalTrending
                 |
             YouTubeDNN
                 |
                 v
             候选融合
                 |
                 v
             DeepFM 精排
                 |
                 v
             多样性重排
                 |
                 v
              推荐结果
```

### 3.2 数据存储

- PostgreSQL：存储用户、电影、评分等核心业务数据。
- Redis：缓存用户画像、历史行为及偏好特征。
- Elasticsearch：提供电影搜索及索引能力。

### 3.3 离线流水线

```text
原始 MovieLens 数据
        |
        v
     特征工程
        |
        +----------+
        |          |
        v          v
    YouTubeDNN   DeepFM
        |          |
        v          v
    电影向量     排序模型
        |          |
        +-----+----+
              |
              v
          模型部署
              |
              v
        Redis 特征上线
```

## 4. 核心算法

### 4.1 多路召回

项目包含四路召回策略：

| 召回通道 | 核心思路 |
|---|---|
| UserPreference | 根据用户历史行为及偏好类型召回 |
| ItemEmbedding | 根据用户历史电影的向量表示检索相似电影 |
| GlobalTrending | 根据全局电影热度进行召回 |
| YouTubeDNN | 利用用户塔和物品塔向量进行相似度检索 |

YouTubeDNN 使用双塔结构，将用户和电影映射到统一向量空间，通过向量相似度获取候选电影。

ItemEmbedding 与 YouTubeDNN 分别承担基于历史物品相似度的召回和用户兴趣向量召回任务。

### 4.2 多路召回融合

原始项目采用 Snake Merge 进行多路候选融合。

在此基础上，实现了可配置的 Weighted Merge（加权轮询融合）。

其核心思想是为不同召回通道分配不同的候选配额，而不是直接对各路召回分数进行加权求和。

当前实验选择：

```python
weights = [4, 0, 6, 0]
```

对应：

- UserPreference：40%
- ItemEmbedding：0%
- GlobalTrending：60%
- YouTubeDNN：0%

融合过程中进行候选去重，并跳过零权重通道的实际召回计算。

### 4.3 DeepFM 精排

使用 DeepFM 对召回候选进行偏好预测。

模型结合：

- FM：建模二阶特征交互。
- DNN：学习高阶非线性特征组合。

离线复现记录：

- Test AUC：0.8388
- gAUC：0.8434

### 4.4 多样性重排

根据电影类型和年代调整候选列表，减少推荐结果过度集中于某一类电影的问题。

## 5. 自主改进与实验

### 5.1 全局时间切分

为避免随机划分导致的时间信息泄露，重新构建全局时间切分实验：

| 数据集 | 交互数量 |
|---|---:|
| Train | 800164 |
| Validation | 100024 |
| Test | 100021 |

使用训练集历史交互不少于 5 次的用户开展评估。

最终有效评估规模：

| 数据集 | 用户数 | 有效交互数 |
|---|---:|---:|
| Validation | 520 | 23639 |
| Test | 1002 | 80298 |

### 5.2 消融实验与融合优化

分别评估四路召回及不同融合策略。

使用验证集选择融合权重，并在测试集上进行最终评估。

测试集结果：

| 策略 | HR@100 | Recall@100 | NDCG@100 |
|---|---:|---:|---:|
| UserPreference | 0.8633 | 0.1813 | 0.1479 |
| ItemEmbedding | 0.7305 | 0.1040 | 0.0961 |
| GlobalTrending | 0.8992 | 0.2184 | 0.2225 |
| YouTubeDNN | 0.6936 | 0.0672 | 0.0672 |
| SnakeMerge | 0.9291 | 0.2149 | 0.1711 |
| WeightedMerge | **0.9411** | **0.2670** | 0.2224 |

相比原始 Snake Merge：

- Recall@100 相对提升 24.24%。
- NDCG@100 相对提升 29.98%。
- HR@100 相对提升 1.29%。

注：GlobalTrending 的 NDCG@100 为 0.2225，略高于 WeightedMerge 的 0.2224。因此，加权融合并非所有单项指标都取得最优结果。

### 5.3 在线召回性能优化

针对在线服务进行了以下改进：

1. 支持根据权重跳过不参与融合的召回通道。
2. 增加历史交互电影过滤。
3. 使用超额召回补充过滤造成的候选损失。
4. 实现可配置的加权轮询融合。
5. 保留原始 Snake Merge 作为对照方案。

单用户预热后执行 10 次测试：

| 指标 | SnakeMerge | WeightedMerge |
|---|---:|---:|
| 平均耗时 | 316.35 ms | 86.45 ms |
| 中位耗时 | 313.89 ms | 84.52 ms |
| 最终候选数量 | 100 | 100 |
| 历史电影重合数 | 0 | 0 |

平均召回耗时降低约 72.67%。

注：该耗时为召回服务测试结果，不代表完整推荐请求的端到端延迟，也不代表高并发场景下的性能。

## 6. 工程改进

在源码复现过程中，完成了以下问题排查和修复：

- 解决 SQLAlchemy 版本兼容问题。
- 修复 Elasticsearch 索引创建过程中的同步/异步调用问题。
- 修复用户偏好召回中的 Elasticsearch 字段错误。
- 修复电影详情页面路由切换后未重新加载数据的问题。
- 修复在线召回过程中的字段错误。
- 完成四路召回的独立测试和在线集成。
- 增加召回融合算法的边界测试。

## 7. 项目部署

### 环境要求

- Python 3.11
- Docker Desktop
- Docker Compose
- Node.js
- uv

### 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

配置：

- FUNREC_RAW_DATA_PATH：原始数据目录。
- FUNREC_PROCESSED_DATA_PATH：处理后数据目录。

### 启动基础设施

```bash
docker compose up --build
```

### 安装后端依赖

```bash
cd backend
uv sync --python 3.11
```

### 数据导入

```bash
make ingest-data-to-database
```

注意：该命令会重建数据库表，不要在已有重要数据的环境中直接执行。

### Elasticsearch 索引

```bash
make index-movies-to-elasticsearch
```

### 离线模型训练及部署

```bash
make run-offline-pipeline
```

### 服务检查

```bash
curl http://localhost:8000/health
```

### 前端访问

```text
http://localhost:3000
```

测试账号：

```text
Email: test@funrec.com
Password: test123456
```

以上为上游项目的标准部署流程。本地 WSL2 开发环境使用的 Compose 覆盖配置需根据实际仓库文件另行配置。

## 8. 项目结构

```text
web_project/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── models/
│   │   └── services/
│   ├── offline/
│   │   ├── feature/
│   │   ├── training/
│   │   └── storage/
│   └── online/
│       ├── cold_start/
│       ├── recall/
│       ├── ranking/
│       └── reranking/
│
├── frontend/
│   └── src/
│       ├── components/
│       ├── views/
│       ├── router/
│       └── services/
│
└── experiments/
    ├── temporal_split.py
    ├── prepare_temporal_data.py
    ├── train_temporal_retrieval.py
    ├── eval_temporal_val.py
    ├── eval_temporal_test.py
    └── RESULTS.md
```

## 9. 当前局限与后续方向

- 当前实验重点集中在召回层，尚未独立验证 DeepFM 精排及多样性重排的增益。
- 冷启动用户尚未纳入本轮时间切分评估。
- 在线性能测试尚未覆盖高并发场景。
- 需要进一步完善 Redis 历史行为与 PostgreSQL 数据的一致性。
- 后续可研究动态召回权重、实时兴趣更新及模型推理优化。

## 10. 致谢

本项目基于 [Datawhale Fun-Rec](https://github.com/datawhalechina/fun-rec) 第 10 章官方电影推荐系统项目进行复现和二次改进。

感谢 Datawhale 及原项目贡献者提供的学习资料和开源代码。

原项目遵循 CC BY-NC-SA 4.0 许可协议，相关衍生成果应遵守原项目许可要求。