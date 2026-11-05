import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(self, emb_dim=16):
        super().__init__()

        # TODO 1：分别定义三个线性层
        # user_emb、short_interest、long_interest
        # 输入维度和输出维度都是 emb_dim
        self.user_linear = nn.Linear(emb_dim, emb_dim, bias=True)
        self.short_linear = nn.Linear(emb_dim, emb_dim, bias=False)
        self.long_linear = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, user_emb, short_interest, long_interest):
        # TODO 2：三个向量分别经过线性变换，再相加
        user_emb_l = self.user_linear(user_emb)
        short_interest_l = self.short_linear(short_interest)
        long_interest_L = self.long_linear(long_interest)

        gate_input = user_emb_l + short_interest_l + long_interest_L

        # TODO 3：使用 sigmoid 得到 gate
        gate = torch.sigmoid(gate_input)

        # TODO 4：按照公式融合长短期兴趣
        final_interest = gate * short_interest + (1 - gate) * long_interest

        return final_interest

if __name__ == "__main__":
    model = GatedFusion(emb_dim=16)

    user_emb = torch.randn(2, 16)
    short_interest = torch.randn(2, 16)
    long_interest = torch.randn(2, 16)

    final_interest = model(
        user_emb, short_interest, long_interest
    )

    print(final_interest.shape)