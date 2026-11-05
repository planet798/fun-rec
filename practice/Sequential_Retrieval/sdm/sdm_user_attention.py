import torch
from torch import nn

class UserAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, user_emb, sequence_output, mask=None):
        # user_emb: [B, D]
        # sequence_output: [B, T, D]

        # 1：调整 user_emb 的维度
        # 提示：unsqueeze(-1)
        user_emb = user_emb.unsqueeze(-1)

        # 2：计算用户向量与每部电影的点积
        # 提示：torch.bmm()
        result = torch.bmm(sequence_output, user_emb)

        # 3：去除最后一个多余维度
        # 提示：squeeze(-1)
        result = result.squeeze(-1)

        if mask is not None:
            masked_result = result.masked_fill(
                ~mask, torch.finfo(result.dtype).min
            )
        else:
            masked_result = result

        atten_weight = torch.softmax(masked_result, dim=-1)

        if mask is not None:
            atten_weight = atten_weight * mask
            atten_weight = atten_weight / (
                atten_weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            )

        # 5：根据注意力权重加权求和
        # 提示：unsqueeze 和 torch.sum()
        short_interest = torch.sum(atten_weight.unsqueeze(-1) * sequence_output, dim=1)

        return short_interest