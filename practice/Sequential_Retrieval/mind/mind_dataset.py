import pandas as pd
import torch

from torch.utils.data import Dataset

DATA_PATH = (
    "/home/dev/projects/MovieLens-Recommender/"
    "data/ml-1m/ratings.dat"
)

def load_sequences(path):
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    # 1：保留 rating >= 4 的正反馈
    df = df[df["rating"] >= 4]

    # 2：按 user_id、timestamp 排序
    df = df.sort_values(
        by=["user_id", "timestamp"],
        ascending = [True, True]
    )

    # 3：按用户分组，将 movie_id 转换为列表
    # 目标：{user_id: [movie1, movie2, ...]}
    sequences = df.groupby("user_id")["movie_id"].apply(list).to_dict()

    return sequences

def build_samples(sequences, max_len=50):
    train_samples = []
    test_samples = []

    for user_id, movies in sequences.items():

        # 1：跳过正反馈少于 3 条的用户
        if len(movies) < 3:
            continue

        # 2：最后一部电影留作测试目标
        train_history = movies[:-1]
        test_item = movies[-1]

        # 3：从训练历史构造多个训练样本
        # (user_id, history, target)
        # history 只保留最近 max_len 个物品
        for i in range(1, len(train_history)):
            history = train_history[:i][-max_len:]
            target = train_history[i]

            train_samples.append((user_id, history, target))

        # 4：构造测试样本
        # (user_id, history, target)
        test_samples.append((
            user_id,
            train_history[-max_len:],
            test_item
        ))

    return train_samples, test_samples

def build_movie_vocab(sequences):
    # 1：提取符合条件的用户训练历史中的所有电影
    train_movies = set()

    for user, movies in sequences.items():
        if len(movies) < 3:
            continue

        # 排除最后一条测试交互
        train_history = movies[:-1]

        # 收集训练历史中的电影并去重
        train_movies.update(train_history)



    # 2：建立 movie_id -> embedding_index 映射
    # 索引从 1 开始，0 留给 Padding
    movie2idx = {
        movie_id: idx
        for idx, movie_id in enumerate(sorted(train_movies), start=1)
    }

    return movie2idx

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    histories, targets = zip(*batch)

    # 1：将历史序列右侧 Padding 到相同长度
    padded_histories = pad_sequence(
        histories,
        batch_first=True,
        padding_value=0
    )

    # 2：生成布尔 Mask，0 为无效位置
    history_mask = padded_histories != 0

    # 3：将 targets 合并成一个张量
    target_ids = torch.stack(targets)

    return padded_histories, history_mask, target_ids


def build_samples_with_val(sequences, max_len=50):
    train_samples = []
    val_samples = []
    test_samples = []

    for user_id, movies in sequences.items():
        if len(movies) < 4:
            continue

        train_history = movies[:-2]
        val_target = movies[-2]
        test_target = movies[-1]

        # 1：利用 train_history 构造训练样本
        # 与之前的滑动历史序列逻辑相同
        for i in range(1, len(train_history)):
            history = train_history[:i][-max_len:]
            target = train_history[i]

            train_samples.append((user_id, history, target))

        # 2：构造验证样本
        # 历史：train_history 的最近 50 部
        # 目标：val_target
        val_samples.append((
            user_id,
            train_history[-max_len:],
            val_target
        ))

        # 3：构造测试样本
        # 历史：movies[:-1] 的最近 50 部
        # 目标：test_target
        test_samples.append((
            user_id,
            movies[:-1][-max_len:],
            test_target
        ))

    return train_samples, val_samples, test_samples


def build_movie_vocab_with_val(sequences):
    train_movies = set()

    for user_id, movies in sequences.items():
        if len(movies) < 4:
            continue

        # 1：只收集 movies[:-2] 中的电影
        train_history = movies[:-2]
        train_movies.update(train_history)

    # 2：建立 movie_id -> index 映射
    # 仍然从 1 开始，0 留给 Padding
    movie2idx = {
        movie2idx : idx
        for idx, movie2idx in enumerate(sorted(train_movies), start=1)
    }

    return movie2idx


class MINDDataset(Dataset):
    def __init__(self, samples, movie2idx):
        self.samples = samples
        self.movie2idx = movie2idx

    def __len__(self):
        # 3：返回样本总数
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, history, target = self.samples[idx]

        # 4：将 history 中的原始电影 ID 转换成索引
        history_ids = [self.movie2idx[mid] for mid in history]

        # 5：将 target 转换成索引
        target_id = self.movie2idx[target]

        return (
            torch.tensor(history_ids, dtype=torch.long),
            torch.tensor(target_id, dtype=torch.long)
        )

if __name__ == "__main__":
    sequences = load_sequences(DATA_PATH)
    train_samples, test_samples = build_samples(sequences)

    movie2idx = build_movie_vocab(sequences)

    dataset = MINDDataset(train_samples, movie2idx)

    print("电影词表大小:", len(movie2idx))
    print("训练样本数:", len(dataset))
    print("第一条样本:", dataset[0])