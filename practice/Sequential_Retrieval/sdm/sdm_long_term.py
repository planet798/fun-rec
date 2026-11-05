import torch
from torch import nn

from sdm_user_attention import UserAttention


class SDMLongEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=16):
        super().__init__()

        # 1：定义电影 Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=0
        )

        # 2：实例化 UserAttention
        self.attention = UserAttention()

    def forward(self, long_history_ids, user_emb):
        # long_history_ids: [B, 40]
        # user_emb: [B, 16]

        # 1：根据 long_history_ids 生成 mask
        # 提示：判断哪些 ID 不等于 0
        mask = long_history_ids != 0

        # 2：调用 UserAttention 时传入 mask

        long_emb = self.embedding(long_history_ids)

        # 4：使用 UserAttention 进行加权聚合
        long_interest = self.attention(user_emb, long_emb, mask)

        return long_interest