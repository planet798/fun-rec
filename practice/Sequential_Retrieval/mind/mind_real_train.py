import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from mind_dataset import (
    DATA_PATH, load_sequences, build_samples_with_val,
    build_movie_vocab_with_val, MINDDataset, collate_fn
)
from capsule import CapsuleLayer
from label_aware import label_aware_attention
from mind_utils import sample_negatives


torch.manual_seed(42)
sequences = load_sequences(DATA_PATH)
train_samples, val_samples, test_samples = build_samples_with_val(sequences)
movie2idx = build_movie_vocab_with_val(sequences)

epochs = 3
D, K = 32, 3
dataset = MINDDataset(train_samples, movie2idx)
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    collate_fn=collate_fn
)

embedding = nn.Embedding(
    num_embeddings= len(movie2idx)+1,
    embedding_dim=D,
    padding_idx=0
)

ROUTING_STD = 0.5
capsule = CapsuleLayer(emb_dim=D, k=K, routing_std=ROUTING_STD)

optimizer = torch.optim.Adam(
    list(embedding.parameters()) + list(capsule.parameters()),
    lr=0.001
)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (history_ids, history_mask, target_ids) in enumerate(loader):

        if (batch_idx + 1) % 500 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(loader)}")

        # 1：重新执行前向传播
        # 包括 Embedding、CapsuleLayer、Attention、负采样
        history_emb = embedding(history_ids)
        target_emb = embedding(target_ids)

        interests = capsule(history_emb, history_mask)

        user_emb = label_aware_attention(interests, target_emb)

        neg_ids = sample_negatives(
            history_ids, target_ids,
            num_items=len(movie2idx),
            n_neg=4
        )

        neg_emb = embedding(neg_ids)


        # 2：计算当前 Batch 的 Loss
        pos_scores = torch.sum(
            user_emb * target_emb,
            dim=-1,
            keepdim=True
        )
        neg_scores = torch.sum(
            user_emb.unsqueeze(1) * neg_emb,
            dim = -1
        )
        logits = torch.cat([pos_scores, neg_scores], dim=1)

        B = history_ids.shape[0]
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)


        optimizer.zero_grad()

        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        # 4：累计 Loss（注意 Batch 大小）
        total_loss += loss.item() * B
        total_samples += B

    # 5：计算当前 Epoch 的平均 Loss
    avg_loss = total_loss / total_samples

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


save_path = (
    Path(__file__).resolve().parent
    / f"mind_val_routing{ROUTING_STD}.pt"
)

torch.save({
    "embedding": embedding.state_dict(),
    "capsule": capsule.state_dict(),
    "movie2idx": movie2idx,
    "D": D,
    "K": K
}, save_path)

print(f"模型已保存：{save_path}")