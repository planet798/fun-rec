import torch

def sample_negatives(
    long_ids,
    short_ids,
    target_ids,
    num_items,
    n_neg=4
):
    # 1：获取 Batch Size
    B = long_ids.shape[0]

    # 2：生成随机负样本
    # 有效电影索引范围 [1, num_items]
    neg_ids = torch.randint(1, num_items + 1, (B, n_neg), device=target_ids.device)

    # 3：合并长短期历史
    all_history = torch.cat([long_ids, short_ids], dim=1)

    # 4：检查负样本是否与历史电影重复
    history_conflict = (
        neg_ids.unsqueeze(-1) == all_history.unsqueeze(1)
    ).any(dim = -1)

    # 5：检查负样本是否与 target 重复
    target_conflict = neg_ids == target_ids.unsqueeze(1)

    invalid = history_conflict | target_conflict

    # 6：对冲突位置重新采样
    # 直到不存在冲突
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
            neg_ids.unsqueeze(-1) == all_history.unsqueeze(1)
        ).any(dim=-1)

        target_conflict = neg_ids == target_ids.unsqueeze(1)

        invalid = history_conflict | target_conflict

    return neg_ids