import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from sdm_dataset import (
    DATA_PATH,
    load_sequences,
    build_samples_with_val,
    build_movie_vocab_with_val,
    SDMDataset,
    collate_fn
)
from sdm_user_tower import SDMUserTower
from sdm_utils import sample_negatives


torch.manual_seed(42)

sequences = load_sequences(DATA_PATH)
train, val, test = build_samples_with_val(sequences)
movie2idx = build_movie_vocab_with_val(sequences)

loader = DataLoader(
    SDMDataset(train, movie2idx),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

model = SDMUserTower(
    num_users=max(sequences.keys()) + 1,
    num_items=len(movie2idx) + 1,
    emb_dim=16,
    num_heads=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

loss_fn = nn.CrossEntropyLoss()


for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_idx, (user_ids, long_ids, short_ids, target_ids) in enumerate(loader):

        # 1：将四个 Tensor 移动到 device
        user_ids = user_ids.to(device)
        long_ids = long_ids.to(device)
        short_ids = short_ids.to(device)
        target_ids = target_ids.to(device)

        # 2：计算用户向量
        user_emb = model(user_ids, long_ids, short_ids)

        # 3：获取正负样本 Embedding
        pos_emb = model.item_embedding(target_ids)

        # 4：计算正负样本得分，拼接 logits
        neg_ids = sample_negatives(long_ids, short_ids, target_ids, len(movie2idx))
        neg_emb = model.item_embedding(neg_ids)

        pos_score = torch.sum(user_emb * pos_emb, dim=-1, keepdim=True)
        neg_score = torch.sum(user_emb.unsqueeze(1) * neg_emb, dim = -1)
        logits = torch.cat([pos_score, neg_score], dim=1)

        # 5：计算 CrossEntropyLoss
        labels = torch.zeros(user_emb.shape[0], dtype=torch.long, device=logits.device)
        loss = loss_fn(logits, labels)

        # 6：清空梯度、反向传播、更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7：累计当前 Batch 的 Loss 和样本数量
        total_loss += loss.item() * user_ids.size(0)
        total_samples += user_ids.size(0)

        if (batch_idx + 1) % 500 == 0:
            print(
                f"Epoch {epoch+1}, "
                f"Batch {batch_idx+1}/{len(loader)}, "
                f"Loss: {loss.item():.4f}"
            )

    # 8：计算并打印当前 Epoch 的平均 Loss
    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# 1：设置模型保存路径
save_path = Path(__file__).resolve().parent / "sdm_model5.pt"

# 2：构造 checkpoint 字典
checkpoint = {
    "model": model.state_dict(),
    "movie2idx": movie2idx,
    "num_users": max(sequences.keys()) + 1,
    "emb_dim": 16,
    "num_heads": 2
}
# 3：调用 torch.save() 保存
torch.save(checkpoint, save_path)