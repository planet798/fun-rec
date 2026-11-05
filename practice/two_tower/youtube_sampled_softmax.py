import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)

# 100部电影，每部电影8维Embedding
item_embedding = nn.Embedding(100, 8)

# 模拟用户塔输出的8维向量
# nn.Parameter表示需要学习的参数
user_vector = nn.Parameter(torch.randn(8))

# 正样本电影ID
target_id = torch.tensor([45])

# 构造负样本候选池，排除电影45和预留ID 0
negative_pool = torch.arange(1, 100)
negative_pool = negative_pool[negative_pool != 45]

# 随机采样4个负样本
indices = torch.randperm(len(negative_pool))[:4]
neg_ids = negative_pool[indices]

# 1. 合并正负样本ID
candidate_ids = torch.cat([target_id, neg_ids])

# 2. 查找候选物品Embedding
candidate_embeddings = item_embedding(candidate_ids)

# 3. 计算用户与5个物品的内积
scores = candidate_embeddings @ user_vector

# 4. 构造正确类别标签
labels = torch.tensor([0])

# 5. 计算交叉熵损失
loss = F.cross_entropy(scores.unsqueeze(0), labels)

optimizer = torch.optim.Adam(
    [user_vector, *item_embedding.parameters()],
    lr=0.01
)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("候选电影ID:", candidate_ids)
print("匹配分数:", scores)
print("Loss:", loss.item())