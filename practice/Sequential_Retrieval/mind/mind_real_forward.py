import torch
from torch import nn
from torch.utils.data import DataLoader

from mind_dataset import (
    DATA_PATH, load_sequences, build_samples,
    build_movie_vocab, MINDDataset, collate_fn
)
from capsule import CapsuleLayer
from label_aware import label_aware_attention

sequences = load_sequences(DATA_PATH)
train_samples, _ = build_samples(sequences)
movie2idx = build_movie_vocab(sequences)

dataset = MINDDataset(train_samples, movie2idx)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

history_ids, history_mask, target_ids = next(iter(loader))

D, K = 32, 3

def sample_negatives(history_ids, target_ids, num_items, n_neg=4):
    B = history_ids.shape[0]

    # 1：创建形状为 [B, n_neg] 的负样本索引
    # 使用 torch.randint()
    # 有效电影索引范围：[1, num_items]

    neg_ids = torch.randint(1, num_items + 1, (B, n_neg), device=history_ids.device)

    # 2：检查负样本是否与历史电影或正样本重复
    # 是否与历史行为重复
    history_conflict = (
        neg_ids.unsqueeze(-1) == history_ids.unsqueeze(1)
    ).any(dim = -1)

    # 是否与正样本重复
    target_conflict = neg_ids == target_ids.unsqueeze(1)

    invalid = history_conflict | target_conflict

    # 3：对冲突的位置重新采样
    while invalid.any():
        count = invalid.sum().item()

        neg_ids[invalid] = torch.randint(
            1,
            num_items + 1,
            (count,),
            device=neg_ids.device
        )

        # 重新计算冲突
        history_conflict = (
            neg_ids.unsqueeze(-1) == history_ids.unsqueeze(1)
        ).any(dim=-1)

        target_conflict = neg_ids == target_ids.unsqueeze(1)

        invalid = history_conflict | target_conflict

    return neg_ids


# 1：创建 nn.Embedding，padding_idx=0
embedding = nn.Embedding(
    num_embeddings= len(movie2idx)+1,
    embedding_dim=D,
    padding_idx=0
)

# 2：创建 CapsuleLayer
capsule = CapsuleLayer(emb_dim=D, k=K)

# 3：将 history_ids 转成 Embedding
history_emb = embedding(history_ids)

# 4：将 target_ids 转成 Embedding
target_emb = embedding(target_ids)

# 5：通过 CapsuleLayer 提取多兴趣向量
interests = capsule(history_emb, history_mask)

# 6：通过 Label-Aware Attention 得到用户向量
user_emb = label_aware_attention(interests, target_emb)

neg_ids = sample_negatives(
    history_ids, target_ids,
    num_items=len(movie2idx),
    n_neg=4
)

# TODO 1：将 neg_ids 转换为 Embedding
neg_emb = embedding(neg_ids)

# TODO 2：计算正样本得分 [B, 1]
pos_scores = torch.sum(
    user_emb * target_emb,
    dim=-1,
    keepdim=True
)

# TODO 3：计算负样本得分 [B, N]
neg_scores = torch.sum(
    user_emb.unsqueeze(1) * neg_emb,
    dim = -1
)

# TODO 4：拼接 logits [B, 1+N]
logits = torch.cat([pos_scores, neg_scores], dim=1)

# TODO 5：生成标签并计算 CrossEntropyLoss
B = history_ids.shape[0]
labels = torch.zeros(B, dtype=torch.long, device=logits.device)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)

# TODO 6：反向传播
optimizer = torch.optim.Adam(
    list(embedding.parameters()) + list(capsule.parameters()),
    lr=0.001
)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("logits:", logits.shape)
print("loss:", loss.item())
print("Embedding梯度:", embedding.weight.grad is not None)
print("Capsule梯度:", capsule.mapping.weight.grad is not None)