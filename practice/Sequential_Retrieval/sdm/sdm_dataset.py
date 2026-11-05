import torch
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import Dataset, DataLoader

DATA_PATH = (
    "/home/dev/projects/MovieLens-Recommender/"
    "data/ml-1m/ratings.dat"
)

def load_sequences(path):
    # 1. 使用 pd.read_csv() 读取 ratings.dat
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    # 2. 保留 rating >= 4 的正反馈
    df = df[df["rating"] >= 4]

    # 3. 按 user_id、timestamp 升序排列
    df = df.sort_values(
        by=["user_id", "timestamp"],
        ascending = [True, True]
    )
    # 4. 按 user_id 分组，将 movie_id 转为列表
    sequences = df.groupby("user_id")["movie_id"].apply(list).to_dict()

    return sequences


def build_movie_vocab_with_val(sequences):
    train_movies = set()

    for user_id, movies in sequences.items():
        # 1. 跳过历史不足 4 条的用户
        if len(movies) < 4:
            continue

        # 2. 只保留 movies[:-2] 中的电影
        train_history = movies[:-2]

        # 3. 将电影加入 train_movies
        train_movies.update(train_history)

    # 4. 构建 movie2idx
    # 从 1 开始编号，0 保留给 Padding
    movie2idx = {
        movie_id : idx
        for idx, movie_id in enumerate(sorted(train_movies), start=1)
    }

    return movie2idx


def split_history(history, short_len=10, long_len=40):
    # 1：取最近 10 部电影作为短期历史
    short_history = history[-short_len:]

    # 2：从剩余历史中取最近 40 部作为长期历史
    long_history = history[-(short_len + long_len):-short_len]

    return long_history, short_history


def build_samples_with_val(sequences):
    train_samples = []
    val_samples = []
    test_samples = []

    for user_id, movies in sequences.items():

        # 1：跳过历史少于 4 条的用户
        if len(movies) < 4:
            continue

        # 2：划分 train_history、val_target、test_target
        # 提示：movies[:-2]、movies[-2]、movies[-1]
        train_history = movies[:-2]
        val_target = movies[-2]
        test_target = movies[-1]

        # 3：使用滑动窗口构造训练样本
        # 每个样本格式：
        # (user_id, long_history, short_history, target)
        for i in range(1, len(train_history)):
            history = train_history[:i]
            target = train_history[i]
            long_history, short_history = split_history(history)

            train_samples.append((user_id, long_history, short_history, target))


        # 4：构造验证样本
        # 使用 train_history 作为历史
        # val_target 作为预测目标
        long_history, short_history = split_history(train_history)

        val_samples.append((
            user_id,
            long_history,
            short_history,
            val_target
        ))

        # 5：构造测试样本
        # 历史使用 movies[:-1]
        # test_target 作为预测目标
        long_history, short_history = split_history(movies[:-1])

        test_samples.append((
            user_id,
            long_history,
            short_history,
            test_target
        ))

    return train_samples, val_samples, test_samples


def collate_fn(batch):
    # 1：使用 zip(*batch) 拆分四种数据
    user_id, long_history, short_history, target = zip(*batch)

    # 2：长期历史右侧 Padding 到 40
    padded_long = torch.stack([
        F.pad(h, (0, 40 - len(h)))
        for h in long_history
    ])

    # 3：短期历史右侧 Padding 到 10
    padded_short = torch.stack([
        F.pad(s, (0, 10 - len(s)))
        for s in short_history
    ])

    # 4：将 user_ids 和 target_ids 组合成 Tensor
    user_ids = torch.stack(user_id)
    target_ids = torch.stack(target)

    # 5：返回四个 Tensor
    return user_ids, padded_long, padded_short, target_ids



class SDMDataset(Dataset):
    def __init__(self, samples, movie2idx):
        self.samples = samples
        self.movie2idx = movie2idx

    def __len__(self):
        # 1：返回样本数量
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, long_history, short_history, target = self.samples[idx]

        long_ids = [
            self.movie2idx[mid]
            for mid in long_history
            if mid in self.movie2idx
        ]

        short_ids = [
            self.movie2idx[mid]
            for mid in short_history
            if mid in self.movie2idx
        ]

        # 4：将 target 转成索引
        target_id = self.movie2idx.get(target, 0)

        # 5：返回四个 torch.long 类型的 Tensor
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(long_ids, dtype=torch.long),
            torch.tensor(short_ids, dtype=torch.long),
            torch.tensor(target_id, dtype=torch.long)
        )


if __name__ == "__main__":
    sequences = load_sequences(DATA_PATH)

    train, val, test = build_samples_with_val(sequences)
    movie2idx = build_movie_vocab_with_val(sequences)

    print("训练样本数:", len(train))
    print("验证样本数:", len(val))
    print("测试样本数:", len(test))
    print("电影词表大小:", len(movie2idx))

    # 1：实例化 SDMDataset
    dataset = SDMDataset(train, movie2idx)
    # 2：创建 DataLoader，batch_size=32
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    # 3：取出第一个 Batch
    user_ids, long_ids, short_ids, target_ids = next(iter(loader))
    # 4：打印四个 Tensor 的 shape
    print(user_ids.shape)
    print(long_ids.shape)
    print(short_ids.shape)
    print(target_ids.shape)