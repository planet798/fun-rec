import torch

from squash import squash
from torch import nn

def route_once(history_emb, k):
    B, T, D = history_emb.shape

    #  1: 初始化路由分数
    # 形状 [B, k, T]
    routing_logits = torch.randn(B, k, T)

    #  2: 计算路由权重
    # 沿历史行为维度进行 Softmax
    weights = torch.softmax(routing_logits, dim=-1)

    #  3: 加权聚合历史行为
    # [B, k, T] @ [B, T, D]
    interest_vectors = torch.matmul(weights, history_emb)

    return interest_vectors


def dynamic_routing(history_emb, history_mask, routing_logits, k, num_iters=3):
    B, T, D = history_emb.shape

    mask = history_mask.unsqueeze(1)

    for i in range(num_iters):

        masked_logits = routing_logits.masked_fill(
            ~mask,
            float("-inf")
        )
        #  2: Softmax 计算权重
        weights = torch.softmax(masked_logits, dim=-1)

        #  3: 矩阵乘法生成兴趣向量
        interest_vectors = torch.matmul(weights, history_emb)

        #  4: squash 压缩兴趣向量
        interest_vectors = squash(interest_vectors)

        #  5: 计算 agreement，形状 [B, K, T]
        agreement = torch.matmul(
            interest_vectors,
            history_emb.transpose(1, 2)
        )
        #  6: 更新 routing_logits
        # 最后一轮不需要更新
        if i < num_iters - 1:
            routing_logits = routing_logits + agreement

    return interest_vectors


class CapsuleLayer(nn.Module):
    def __init__(self, emb_dim, k, num_iters=3, routing_std=0.01):
        super().__init__()

        self.register_buffer(
            "routing_init",
            torch.randn(1, k, 50) * routing_std
        )

        self.k = k
        self.num_iters = num_iters

        #  1：创建可训练的映射矩阵
        # 使用 nn.Linear(emb_dim, emb_dim, bias=False)
        self.mapping = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, history_emb, history_mask):

        #  2：将历史行为经过映射层
        mapped_emb = self.mapping(history_emb)

        B, T, D = mapped_emb.shape

        routing_logits = self.routing_init[:, :, :T]
        routing_logits = routing_logits.expand(B, -1, -1).clone()

        #  3：调用已经实现的 dynamic_routing()
        interest_vectors = dynamic_routing(mapped_emb, history_mask, routing_logits, self.k, self.num_iters)

        return interest_vectors

if __name__ == "__main__":
    torch.manual_seed(42)

    model = CapsuleLayer(emb_dim=8, k=3)

    history_emb = torch.randn(2, 5, 8)
    history_mask = torch.tensor([
        [True, True, True, True, True],
        [True, True, True, False, False]
    ])

    output = model(history_emb, history_mask)

    print("输出形状:", output.shape)
    print("向量模长:", torch.linalg.vector_norm(output, dim=-1))

    loss = output.sum()
    loss.backward()

    print("映射矩阵是否有梯度:",
        model.mapping.weight.grad is not None)