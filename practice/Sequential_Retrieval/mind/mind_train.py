import torch
from torch import nn

from capsule import CapsuleLayer
from label_aware import label_aware_attention

torch.manual_seed(42)

B, T, D, K, N = 2, 5, 8, 3, 4

# 模拟训练数据
history_emb = torch.randn(B, T, D)

history_mask = torch.tensor([
    [True, True, True, True, True],
    [True, True, True, False, False]
])

pos_emb = torch.randn(B, D)
neg_emb = torch.randn(B, N, D)

# 1. 提取多兴趣向量
capsule = CapsuleLayer(emb_dim=D, k=K)

optimizer = torch.optim.Adam(capsule.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 创建标签
# 所有正样本均位于第 0 列
labels = torch.zeros(B, dtype=torch.long)

for epoch in range(101):
    optimizer.zero_grad()

    interest_vectors = capsule(history_emb, history_mask)

    # 2. 根据正样本融合兴趣
    user_emb = label_aware_attention(
        interest_vectors, pos_emb
    )

    # 计算正样本得分
    # 输出 [B, 1]
    pos_scores = torch.sum(
        user_emb * pos_emb,
        dim=-1,
        keepdim=True
    )

    # 计算负样本得分
    # 输出 [B, N]
    neg_scores = torch.sum(
        user_emb.unsqueeze(1) * neg_emb,
        dim=-1,
    )

    # 拼接正负样本得分
    # 输出 [B, 1 + N]
    logits = torch.cat([pos_scores, neg_scores], dim=1)

    loss = loss_fn(logits, labels)

    loss.backward()
    optimizer.step()


    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")