import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from collections import Counter

path = "/home/dev/projects/MovieLens-Recommender/data/ml-1m/ratings.dat"

df = pd.read_csv(
    path,
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

df = df[df["rating"] >= 4]
df = df.sort_values(["user_id", "timestamp"])

train_histories = []
train_user_ids = []
test_items = {}
grouped = df.groupby("user_id")["movie_id"].agg(list)

for user_id, movie_ids in grouped.items():
    movie_ids = [str(x) for x in movie_ids]

    if len(movie_ids) < 2:
        continue

    train_histories.append(movie_ids[:-1])
    train_user_ids.append(user_id)
    test_items[user_id] = movie_ids[-1]

model = Word2Vec.load("practice/item2vec/item2vec_ml1m.model")

total_hit = 0
warm_user = 0
warm_hit = 0

for user_id, history in zip(train_user_ids, train_histories):

    vectors = [model.wv[item] for item in history]
    user_vector = np.mean(vectors, axis=0)

    candidates = model.wv.most_similar(
        positive=[user_vector],
        topn = len(set(history)) + 10
    )

    # movie_top10
    history_set = set(history)

    movie_top10 = [
        item
        for item, score in candidates
        if item not in history_set
    ][:10]

    if test_items[user_id] in movie_top10:
        total_hit += 1

    if test_items[user_id] in model.wv.key_to_index:
        # Warm Item
        warm_user += 1
        if test_items[user_id] in movie_top10:
            warm_hit += 1


total_users = len(test_items)

hr10 = total_hit / total_users
warm_hr10 = warm_hit / warm_user if warm_user > 0 else 0
cold_rate = (total_users - warm_user) / total_users

# print("HR@10:", hr10)
# print("Warm HR@10:", warm_hr10)
# print("Cold Item Rate:", cold_rate)


# 1. 统计训练集中每部电影的热度
item_counts = Counter()

for history in train_histories:
    item_counts.update(history)

# 2. 按热度降序排列
popular_items = [
    item for item, count in item_counts.most_common()
]

# 3. 生成推荐并计算 HR@10
popularity_hit = 0

for user_id, history in zip(train_user_ids, train_histories):
    history_set = set(history)

    # 排除已交互电影，取 Top-10
    movie_top10 = [
        item for item in popular_items
        if item not in history_set
    ][:10]

    if test_items[user_id] in movie_top10:
        popularity_hit += 1

# 4. 输出对比结果
popularity_hr10 = popularity_hit / len(test_items)

print("\n===== 模型效果对比 =====")
print(f"Item2Vec HR@10:    {hr10:.4%}")
print(f"Popularity HR@10: {popularity_hr10:.4%}")
