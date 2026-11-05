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

# 构造一个负样本电影
neg_item_features = torch.randn(2, 8)
neg_input = neg_item_features.reshape(-1)

neg_embedding = F.normalize(
    item_tower(neg_input), p=2, dim=0
)

neg_score = torch.dot(user_embedding, neg_embedding)

# 合并正负样本得分
scores = torch.stack([score, neg_score])

# 构造标签：正样本1，负样本0
labels = torch.tensor([1.0, 0.0])

# 计算BCE损失
# 这里使用WithLogits版本，内部包含Sigmoid
loss = F.binary_cross_entropy_with_logits(
    scores, labels
)

# 定义优化器
optimizer = torch.optim.Adam(
    list(user_tower.parameters()) +
    list(item_tower.parameters()),
    lr=0.001
)

# 梯度清零、反向传播、参数更新
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())