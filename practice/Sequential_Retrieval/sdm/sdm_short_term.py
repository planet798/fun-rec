import torch

from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)
from torch import nn

class ShortTermEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=16):
        super().__init__()

        # 1: 定义电影 Embedding 层
        self.embedding = nn.Embedding(
            num_embeddings= vocab_size,
            embedding_dim= emb_dim,
            padding_idx= 0
        )

        # 2: 定义 LSTM
        # input_size = emb_dim
        # hidden_size = emb_dim
        # batch_first = True
        self.lstm = nn.LSTM(
            input_size= emb_dim,
            hidden_size= emb_dim,
            batch_first= True
        )

    def forward(self, history_ids):
        history_emb = self.embedding(history_ids)

        # 1：计算每个用户的有效历史长度
        val_len = (history_ids != 0).sum(dim=1)

        # 2：处理长度为 0 的用户
        # 提示：pack 不允许长度为 0
        # 可以将用于 pack 的长度最小限制为 1
        safe_lengths = val_len.clamp_min(1)

        # 3：使用 pack_padded_sequence
        packed = pack_padded_sequence(
            history_emb,
            safe_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # 4：将 PackedSequence 输入 LSTM
        packed_output, _ = self.lstm(packed)

        # 5：使用 pad_packed_sequence 还原
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=history_ids.size(1)
        )

        # 6：将原本长度为 0 的序列输出清零
        # 同时确保其他 Padding 位置也是零
        empty_mask = (val_len == 0)
        sequence_output = output.masked_fill(
            empty_mask[:, None, None],
            0.0
        )
        return sequence_output