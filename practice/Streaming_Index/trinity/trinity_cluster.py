"""Trinity 练习 1:将物品 Embedding 分配到最近的聚类中心。"""

import torch


def assign_clusters(item_embs: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Args:
        item_embs: [N, D],N 个物品的向量。
        centroids: [K, D],K 个聚类中心。
    Returns:
        cluster_ids: [N],每个物品所属的聚类编号(0 至 K-1)。
    """
    # TODO 1：用广播计算所有物品与所有聚类中心的平方欧氏距离,得到 [N, K]。
    # 提示：先通过 unsqueeze 增加维度,再相减、平方,沿最后一维求和。

    # TODO 2：从 [N, K] 中取得距离最小的聚类编号,得到 [N]。
    raise NotImplementedError("请完成 TODO 1 和 TODO 2")


if __name__ == "__main__":
    item_embs = torch.tensor([[0.9, 0.1], [0.1, 0.8], [-0.9, 0.0], [0.8, 0.2]])
    centroids = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    cluster_ids = assign_clusters(item_embs, centroids)
    print("cluster_ids:", cluster_ids.tolist())
    assert cluster_ids.tolist() == [0, 1, 2, 0]
    print("练习 1 通过")
