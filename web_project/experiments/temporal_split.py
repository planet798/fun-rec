import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

raw = Path(os.environ["FUNREC_RAW_DATA_PATH"])

ratings = pd.read_pickle(raw / "ratings.pkl")
ratings["timestamp"] = pd.to_numeric(ratings["timestamp"])

# 按时间排序
ratings = ratings.sort_values("timestamp")

# 确定时间切分点
t1 = ratings["timestamp"].quantile(0.8)
t2 = ratings["timestamp"].quantile(0.9)

# 划分数据集
train = ratings[ratings["timestamp"] < t1]

val = ratings[
    (ratings["timestamp"] >= t1) &
    (ratings["timestamp"] < t2)
]

test = ratings[ratings["timestamp"] >= t2]

# 统计数据
for name, data in [
    ("Train", train),
    ("Val", val),
    ("Test", test)
]:
    print(f"\n{name}")
    print("交互数量:", len(data))
    print("用户数量:", data["user_id"].nunique())
    print("电影数量:", data["movie_id"].nunique())
    print("时间范围:", pd.to_datetime(
        data["timestamp"], unit="s"
    ).agg(["min", "max"]))

# 检查冷启动用户
train_users = set(train["user_id"])

for name, data in [("Val", val), ("Test", test)]:
    new_users = set(data["user_id"]) - train_users
    print(f"{name} 未见用户数量:", len(new_users))

# 统计训练集中每个用户的交互次数
user_counts = train.groupby("user_id").size()

# 筛选历史交互不少于5次的用户
warm_users = set(
    user_counts[user_counts >= 5].index
)

# 统计模型训练阶段未见过的电影
train_movies = set(train["movie_id"])

for name, data in [("Val", val), ("Test", test)]:

    warm_data = data[
        data["user_id"].isin(warm_users)
    ]

    print(f"\n{name} 有效数据")
    print("有效用户数:", warm_data["user_id"].nunique())
    print("有效交互数:", len(warm_data))
    print("未见电影数:", len(
        set(warm_data["movie_id"]) - train_movies
    ))


# 创建独立实验数据目录
output_dir = Path(
    "../tmp/web_project/temporal_experiment"
)
output_dir.mkdir(parents=True, exist_ok=True)

# 保存训练集
train.to_pickle(output_dir / "train_ratings.pkl")

# 保存正常用户验证集和测试集
val_warm = val[val["user_id"].isin(warm_users)]
test_warm = test[test["user_id"].isin(warm_users)]

val_warm.to_pickle(output_dir / "val_ratings.pkl")
test_warm.to_pickle(output_dir / "test_ratings.pkl")

print("数据集保存完成:", output_dir.resolve())