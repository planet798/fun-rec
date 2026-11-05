import torch
from torch import nn

from sdm_short_encoder import SDMShortEncoder
from sdm_long_term import SDMLongEncoder
from sdm_gate import GatedFusion


class SDMUserTower(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=16, num_heads=2):
        super().__init__()

        # 共享电影 Embedding
        self.item_embedding = nn.Embedding(
            num_items,
            emb_dim,
            padding_idx=0
        )
        # 1：定义用户 Embedding 层
        # 注意：这里是用户 ID，不是电影 ID
        self.useremb = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=emb_dim,
            padding_idx=0
        )

        # 2：实例化短期兴趣编码器
        self.short_interest = SDMShortEncoder(num_items, emb_dim, num_heads)
        self.short_interest.short_term.embedding = self.item_embedding

        # 3：实例化长期兴趣编码器
        self.long_interest = SDMLongEncoder(num_items, emb_dim)
        self.long_interest.embedding = self.item_embedding

        # 4：实例化 GatedFusion
        self.gatefusion = GatedFusion(emb_dim)

    def forward(self, user_ids, long_history_ids, short_history_ids):

        # 5：将 user_ids 转换为 user_emb
        user_emb = self.useremb(user_ids)

        # 6：计算短期兴趣向量
        short_interest = self.short_interest(short_history_ids, user_emb)

        # 7：计算长期兴趣向量
        long_interest = self.long_interest(long_history_ids, user_emb)

        # 8：门控融合，得到最终用户向量
        final_user_emb = self.gatefusion(
            user_emb, short_interest, long_interest
        )

        return final_user_emb