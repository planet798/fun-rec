import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)

user_features = torch.randn(4, 8)
item_features = torch.randn(2, 8)

# 1. 构建用户塔
user_tower = nn.Sequential(
    nn.Linear(32, 128),
    nn.Tanh(),
    nn.Linear(128, 64),
    nn.Tanh(),
    nn.Linear(64, 32)
)

# 2. 构建物品塔
item_tower = nn.Sequential(
    nn.Linear(16, 128),
    nn.Tanh(),
    nn.Linear(128, 64),
    nn.Tanh(),
    nn.Linear(64, 32)
)


# 3. 将用户和物品特征分别展平为一维
user_input = user_features.reshape(-1)
item_input = item_features.reshape(-1)

user_output = user_tower(user_input)
item_output = item_tower(item_input)

# 4. DNN 前向传播并做L2归一化
user_embedding = F.normalize(user_output, p=2, dim=0)
item_embedding = F.normalize(item_output, p=2, dim=0)

# 5. 计算匹配分数
score = torch.dot(user_embedding, item_embedding)

print("用户向量形状:", user_embedding.shape)
print("物品向量形状:", item_embedding.shape)
print("匹配分数:", score)