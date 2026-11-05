import torch

def squash(x, eps=1e-9):
    # x: [batch_size, K, embedding_dim]

    # 1: 计算向量模长的平方
    norm_sq = torch.sum(torch.square(x), dim=-1, keepdim=True)

    # 2: 计算 squash 缩放系数
    scale = norm_sq / (1 + norm_sq)
    direction = x / torch.sqrt(norm_sq + eps) # eps防止分母为0

    # 3: 返回压缩后的向量
    return scale * direction

if __name__ == '__main__':
    x = torch.randn(2, 3, 8)
    output = squash(x)

    print("输出形状:", output.shape)
    print("向量模长:", torch.linalg.vector_norm(output, dim=-1))