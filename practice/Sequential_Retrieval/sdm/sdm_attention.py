import torch
from torch import nn

class SDMAttention(nn.Module):
    def __init__(self, emb_dim=16, num_heads=2):
        super().__init__()

        # 1: 定义 LayerNorm
        self.layer_norm = nn.LayerNorm(emb_dim)

        # 2: 定义 nn.MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, sequence_output, mask):

        norm_output = self.layer_norm(sequence_output)
        # 1：如果提供 mask，将其取反
        key_padding_mask = ~mask

        # 避免所有 Key 都被屏蔽导致 NaN
        all_padding = key_padding_mask.all(dim=1)

        safe_key_padding_mask = key_padding_mask.clone()
        safe_key_padding_mask[all_padding, 0] = False

        # 2：调用 nn.MultiheadAttention
        attention_output, _ = self.attention(
            norm_output,
            norm_output,
            norm_output,
            key_padding_mask=safe_key_padding_mask,
            need_weights=False
        )

        # 3：将 Padding 对应的输出位置清零
        attention_output = attention_output.masked_fill(
            ~mask.unsqueeze(-1),
            0.0
        )

        return attention_output