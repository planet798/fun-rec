import torch
from pathlib import Path

from sdm_user_tower import SDMUserTower
from sdm_dataset import (
    DATA_PATH,
    load_sequences,
    build_samples_with_val,
    SDMDataset,
    collate_fn
)

@torch.no_grad()
def recommend_top10(user_id, long_history, short_history):

    # 1：将电影 ID 映射成 Embedding 索引
    # 1. 原始电影 ID 转为 Embedding 索引
    long_ids = [movie2idx[mid] for mid in long_history if mid in movie2idx]
    short_ids = [movie2idx[mid] for mid in short_history if mid in movie2idx]

    # 2. 右侧 Padding
    long_ids = long_ids + [0] * (40 - len(long_ids))
    short_ids = short_ids + [0] * (10 - len(short_ids))

    # 3. 转换成 Tensor，并增加 Batch 维度
    user_ids = torch.tensor([user_id], dtype=torch.long)
    long_ids = torch.tensor([long_ids], dtype=torch.long)
    short_ids = torch.tensor([short_ids], dtype=torch.long)

    # 2：调用 SDMUserTower
    # 得到 user_emb：[1, 16]
    user_emb = model(user_ids, long_ids, short_ids)

    # 3：计算所有电影的内积得分
    # scores：[1, num_items]
    # 1. 获取所有电影的 Embedding
    item_emb = model.item_embedding.weight

    # 2. 计算用户对所有电影的得分
    scores = user_emb @ item_emb.T

    # 4：屏蔽 Padding ID=0以及已经观看过的电影
    # 1. 屏蔽 Padding ID=0
    scores[:, 0] = float("-inf")

    # 2. 合并长期和短期历史
    history_ids = torch.cat([long_ids, short_ids], dim=1)

    # 3. 屏蔽已经观看过的电影
    scores[:, history_ids.squeeze(0)] = float("-inf")

    # 5：使用 torch.topk() 选出 10 部电影
    top10_indices = torch.topk(scores, k=10, dim=-1).indices.squeeze(0)

    return top10_indices.tolist()


checkpoint_path = Path(__file__).resolve().parent / "sdm_model5.pt"

# 1：使用 torch.load() 加载 checkpoint
ckpt = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)
# 2：读取 movie2idx、num_users、emb_dim、num_heads
movie2idx = ckpt["movie2idx"]
num_users = ckpt["num_users"]
emb_dim = ckpt["emb_dim"]
num_heads = ckpt["num_heads"]

# 3：实例化 SDMUserTower
# num_items = len(movie2idx) + 1
model = SDMUserTower(
    num_users=num_users,
    num_items=len(movie2idx) + 1,
    emb_dim=emb_dim,
    num_heads=num_heads
)

# 4：使用 load_state_dict() 恢复权重
model.load_state_dict(ckpt["model"])

# 5：调用 model.eval()
model.eval()

sequences = load_sequences(DATA_PATH)
_, val_samples, _ = build_samples_with_val(sequences)

user_id, long_history, short_history, target = val_samples[0]

top10 = recommend_top10(
    user_id, long_history, short_history
)

# 1：初始化统计变量
total = len(val_samples)
hits = 0
warm_users = 0
cold_users = 0

# 2：遍历验证集
for user_id, long_history, short_history, target in val_samples:

    # 3：判断 target 是否在 movie2idx 中
    # 如果不在，cold_users 加 1，并跳过
    if target not in movie2idx:
        cold_users += 1
        continue

    # 4：warm_users 加 1
    warm_users += 1

    # 5：调用 recommend_top10()
    recommendations = recommend_top10(user_id, long_history, short_history)

    # 6：将 target 转换为 Embedding 索引
    target_idx = movie2idx[target]

    # 7：判断目标电影是否命中 Top-10
    # 命中则 hits 加 1
    if target_idx in recommendations:
        hits += 1

# 8：计算 HR@10 和 Warm HR@10
hr10 = hits / total
warm_hr10 = hits / warm_users if warm_users > 0 else 0.0

# 9：打印统计结果
print(f"Val HR@10: {hr10:.4%}")
print(f"Val Warm HR@10: {warm_hr10:.4%}")
print(f"Hits: {hits}/{total}")
print(f"Cold Users: {cold_users}")