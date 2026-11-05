import torch

def label_aware_attention(interest_vectors, target_emb):
    # interest_vectors: [B, K, D]
    # target_emb: [B, D]

    # TODO 1: 将 target_emb 扩展为 [B, 1, D]
    target = target_emb.unsqueeze(1)

    # TODO 2: 计算每个兴趣与目标物品的点积
    # 得到 [B, K]
    scores = torch.sum(interest_vectors * target, dim=-1)

    # TODO 3: 沿兴趣维度执行 Softmax
    weights = torch.softmax(scores, dim=-1)

    # TODO 4: 对兴趣向量加权求和
    # 输出 [B, D]
    user_emb = torch.sum(
        interest_vectors * weights.unsqueeze(-1),
        dim=1
    )

    return user_emb


if __name__ == "__main__":
    interests = torch.randn(2, 3, 8)
    target = torch.randn(2, 8)

    output = label_aware_attention(interests, target)
    print(output.shape)