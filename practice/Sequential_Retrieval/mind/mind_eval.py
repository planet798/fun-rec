import torch
from torch import nn
from pathlib import Path

from capsule import CapsuleLayer
from mind_dataset import (
    DATA_PATH, load_sequences, build_samples_with_val
)

@torch.no_grad()
def recommend_top10(history):
    # 1：复用已经实现的电影 ID 转换、Mask
    history_ids = torch.tensor(
        [movie2idx.get(mid, 0) for mid in history],
        dtype=torch.long
    ).unsqueeze(0)

    history_mask = history_ids != 0
    # 2：计算 3 个兴趣向量
    history_emb = embedding(history_ids)
    interests = capsule(history_emb, history_mask)

    # 3：计算得分、过滤历史电影
    item_emb = embedding.weight

    scores = interests[0] @ item_emb.T

    scores[:, 0] = float("-inf")
    scores[:, history_ids.squeeze(0)] = float("-inf")

    top_indices = torch.topk(scores, 10, dim=-1).indices

    # 4：Top-K、合并、去重、排序
    all_indices = top_indices.flatten()

    candidate_ids = torch.unique(all_indices)

    candidate_scores = scores[:, candidate_ids].max(dim=0).values

    top_positions = torch.topk(candidate_scores, 10).indices
    final_indices = candidate_ids[top_positions]

    return final_indices.tolist()


# 1. 加载模型文件
checkpoint_path = (
    Path(__file__).resolve().parent
    / "mind_val_routing0.5.pt"
)

ckpt = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

movie2idx = ckpt["movie2idx"]
D, K = ckpt["D"], ckpt["K"]

# 2. 创建模型结构
embedding = nn.Embedding(
    len(movie2idx) + 1,
    D,
    padding_idx=0
)

capsule = CapsuleLayer(emb_dim=D, k=K)

# 1：使用 load_state_dict() 恢复两个模块的权重
embedding.load_state_dict(ckpt["embedding"])
capsule.load_state_dict(ckpt["capsule"])

# 2：将两个模块切换到 eval() 模式
embedding.eval()
capsule.eval()

# 3. 获取测试数据
sequences = load_sequences(DATA_PATH)
_, val_samples, test_samples = build_samples_with_val(sequences)

eval_samples = val_samples

total = len(eval_samples)
hits = 0
warm_users = 0
cold_users = 0

for user_id, history, target in eval_samples:

    if target not in movie2idx:
        cold_users += 1
        continue

    warm_users += 1

    # 1：调用 recommend_top10(history)
    recommendations = recommend_top10(history=history)

    # 2：将 target 转换为 Embedding 索引
    target_idx = movie2idx[target]

    # 3：判断 target_idx 是否在推荐结果中
    if target_idx in recommendations:
        hits += 1

# 4：计算两个指标
hr10 = hits / total
warm_hr10 = hits / warm_users if warm_users > 0 else 0.0

# test
import torch.nn.functional as F

cosine_values = []
same_top10 = 0

for user_id, history, target in eval_samples[:100]:

    history_ids = torch.tensor(
        [movie2idx.get(mid, 0) for mid in history],
        dtype=torch.long
    ).unsqueeze(0)

    with torch.no_grad():
        interests = capsule(
            embedding(history_ids),
            history_ids != 0
        )[0]  # [3, 32]

        # 1：计算兴趣 0 和兴趣 1 的余弦相似度
        cosine = F.cosine_similarity(
            interests[0], interests[1],
            dim=0
        ).item()

        # 2：分别计算三个兴趣的 Top-10 电影
        scores = interests @ embedding.weight.T

        # 屏蔽 Padding 和历史电影
        scores[:, 0] = float("-inf")
        scores[:, history_ids.squeeze(0)] = float("-inf")

        # 每个兴趣取 Top-10
        top_indices = torch.topk(scores, k=10, dim=-1).indices

        # 3：判断三个兴趣的 Top-10 是否完全相同
        is_same = (
            torch.equal(top_indices[0], top_indices[1])
            and torch.equal(top_indices[1], top_indices[2])
        )

        if is_same:
            same_top10 += 1

    cosine_values.append(cosine)

# 4：计算平均余弦相似度和 Top-10 完全重合比例
avg_cosine = sum(cosine_values) / len(cosine_values)
same_ratio = same_top10 / len(cosine_values)

print(f"Val HR@10: {hr10:.4%}")
print(f"Val Warm HR@10: {warm_hr10:.4%}")
print(f"Hits: {hits}/{total}, Cold: {cold_users}")