import torch
from torch import nn

torch.manual_seed(42)

# 假设电影库共有100部电影
# 每部电影Embedding为8维
item_embedding = nn.Embedding(
    100, 8, padding_idx=0
)

# 用户历史观看电影ID
# 0表示补齐位置，并非真实电影
history_ids = torch.tensor([12, 25, 37, 0, 0])

# 查找对应的Embedding
hist_embeddings = item_embedding(history_ids)

def mean_pooling(history_ids, hist_embeddings):

    # 1. 找出有效的历史电影（ID不为0）
    mask = history_ids != 0

    # 2. 取出有效电影的Embedding
    valid_embeddings = hist_embeddings[mask]

    # 3. 沿历史长度维度求平均
    history_vector = valid_embeddings.mean(dim = 0)

    return history_vector

history_vector = mean_pooling(
    history_ids, hist_embeddings
)



import torch.nn.functional as F

# 模拟4个用户特征，每个Embedding为8维
user_features = torch.randn(4, 8)

# 展平用户特征
user_flat = user_features.reshape(-1)

# 将用户特征与历史兴趣向量拼接
user_input = torch.cat([user_flat, history_vector])

# 构建用户DNN
user_tower = nn.Sequential(
    nn.Linear(40, 32),
    nn.ReLU(),
    nn.Linear(32, 8)
)

# 前向传播 + L2归一化
user_vector = F.normalize(
    user_tower(user_input), p=2, dim=0
)

# 查找目标电影ID=45的Embedding并归一化
target_id = torch.tensor(45)
item_vector = F.normalize(
    item_embedding(target_id), p=2, dim=0
)

# 计算匹配分数
score = torch.dot(user_vector, item_vector)

print("用户输入形状:", user_input.shape)
print("用户向量形状:", user_vector.shape)
print("物品向量形状:", item_vector.shape)
print("匹配分数:", score)