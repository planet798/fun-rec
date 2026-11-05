import torch

def sample_negatives(history_ids, target_ids, num_items, n_neg=4):
    B = history_ids.shape[0]

    # 1：创建形状为 [B, n_neg] 的负样本索引
    # 使用 torch.randint()
    # 有效电影索引范围：[1, num_items]

    neg_ids = torch.randint(1, num_items + 1, (B, n_neg), device=history_ids.device)

    # 2：检查负样本是否与历史电影或正样本重复
    # 是否与历史行为重复
    history_conflict = (
        neg_ids.unsqueeze(-1) == history_ids.unsqueeze(1)
    ).any(dim = -1)

    # 是否与正样本重复
    target_conflict = neg_ids == target_ids.unsqueeze(1)

    invalid = history_conflict | target_conflict

    # 3：对冲突的位置重新采样
    while invalid.any():
        count = invalid.sum().item()

        neg_ids[invalid] = torch.randint(
            1,
            num_items + 1,
            (count,),
            device=neg_ids.device
        )

        # 重新计算冲突
        history_conflict = (
            neg_ids.unsqueeze(-1) == history_ids.unsqueeze(1)
        ).any(dim=-1)

        target_conflict = neg_ids == target_ids.unsqueeze(1)

        invalid = history_conflict | target_conflict

    return neg_ids