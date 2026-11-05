import torch
from torch import nn

from sdm_short_term import ShortTermEncoder
from sdm_attention import SDMAttention
from sdm_user_attention import UserAttention


class SDMShortEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, num_heads=2):
        super().__init__()

        self.short_term = ShortTermEncoder(vocab_size, emb_dim)
        self.attention = SDMAttention(emb_dim, num_heads)
        self.user_attention = UserAttention()

    def forward(self, history_ids, user_emb):
        mask = (history_ids != 0)
        # 4：history_ids 输入 LSTM
        sequence_output = self.short_term(history_ids)

        # 5：LSTM 输出输入 Multi-Head Attention
        attention_output = self.attention(sequence_output, mask)

        # 6：结合 user_emb 计算短期兴趣向量
        short_interest = self.user_attention(user_emb, attention_output, mask)

        return short_interest


if __name__ == "__main__":
    model = SDMShortEncoder(
        vocab_size=4000,
        emb_dim=16,
        num_heads=2
    )

    history_ids = torch.randint(1, 4000, (2, 10))
    user_emb = torch.randn(2, 16)

    short_interest = model(history_ids, user_emb)
    print(short_interest.shape)