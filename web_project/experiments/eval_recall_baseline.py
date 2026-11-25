
import os
import pickle
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from funrec.models.layers import DNNs, L2NormalizeLayer

K = 100
N_USERS = 6040

# 1. 加载数据
load_dotenv(".env")

raw = Path(os.environ["FUNREC_RAW_DATA_PATH"])
tmp = Path(os.environ["FUNREC_PROCESSED_DATA_PATH"]) / "web_project"

ratings = pd.read_pickle(raw / "ratings.pkl")
movies = pd.read_pickle(raw / "movies.pkl")

with open(tmp / "vocab_dict.pkl", "rb") as f:
    vocab = pickle.load(f)

with open(tmp / "train_eval_sample_final.pkl", "rb") as f:
    test = pickle.load(f)["test"]

for col in ["user_id", "movie_id"]:
    ratings[col] = ratings[col].astype(str)

movies["movie_id"] = movies["movie_id"].astype(str)

user_ids = np.asarray(vocab["user_id"]).astype(str)
movie_ids = np.asarray(vocab["movie_id"]).astype(str)

# 2. 找到所有测试答案，并从评分记录中排除
test_users = user_ids[test["user_id"] - 1]
test_targets = movie_ids[test["movie_id"] - 1]

holdout = pd.MultiIndex.from_arrays(
    [test_users, test_targets]
)

rating_pairs = pd.MultiIndex.from_frame(
    ratings[["user_id", "movie_id"]]
)

mask = rating_pairs.isin(holdout)
assert mask.sum() == len(test_targets)

train = ratings.loc[~mask].copy()

print("测试交互数量:", mask.sum())
print("剩余历史交互:", len(train))

# 3. 仅使用历史数据计算电影热度
stats = train.groupby("movie_id")["rating"].agg(
    avg_rating="mean",
    rating_count="count"
).reset_index()

movies = movies.merge(stats, on="movie_id", how="left")
movies["startYear"] = pd.to_numeric(
    movies["startYear"], errors="coerce"
)

genre_map = {
    row.movie_id: set(str(row.genres).split("|"))
    for row in movies.itertuples()
}

popular = movies[
    (movies["avg_rating"] >= 5) &
    (movies["rating_count"] >= 50)
].sort_values(
    ["rating_count", "avg_rating"],
    ascending=False
)["movie_id"].tolist()[:K]

preference_pool = movies[
    movies["avg_rating"] >= 5
].sort_values(
    ["startYear", "avg_rating"],
    ascending=False
)["movie_id"].tolist()

histories = train.sort_values("timestamp").groupby(
    "user_id"
)["movie_id"].agg(list).to_dict()

# 4. 加载双塔模型与电影向量
deploy = tmp / "deployed_models"

embeddings = np.load(deploy / "item_embeddings.npy")
saved_ids = np.load(deploy / "movie_ids.npy").astype(str)

assert np.array_equal(saved_ids, movie_ids)

user_model = tf.keras.models.load_model(
    tmp / "saved_models" / "user_model",
    custom_objects={
        "DNNs": DNNs,
        "L2NormalizeLayer": L2NormalizeLayer
    },
    compile=False
)

# 5. 蛇形融合
def snake_merge(channels, k):
    result = []
    seen = set()
    pointers = [0] * len(channels)

    order = list(range(len(channels)))
    order += order[::-1]

    while len(result) < k:
        if all(p >= len(c) for p, c in zip(pointers, channels)):
            break

        for i in order:
            if pointers[i] >= len(channels[i]):
                continue

            mid = channels[i][pointers[i]]
            pointers[i] += 1

            if mid not in seen:
                result.append(mid)
                seen.add(mid)

            if len(result) == k:
                return result

    return result


def weighted_merge(channels, weights, k):
    result = []
    seen = set()
    pointers = [0] * len(channels)

    while len(result) < k:
        added = False

        for i, weight in enumerate(weights):
            count = 0

            while count < weight and pointers[i] < len(channels[i]):
                mid = channels[i][pointers[i]]
                pointers[i] += 1

                if mid in seen:
                    continue

                result.append(mid)
                seen.add(mid)

                count += 1
                added = True

                if len(result) >= k:
                    return result

        if not added:
            break

    return result

# 6. 随机抽取测试用户
rng = np.random.default_rng(42)
indices = rng.choice(
    len(test_users), N_USERS, replace=False
)

input_keys = [
    "user_id", "gender", "age", "occupation",
    "zip_code", "hist_movie_id", "hist_genres"
]

inputs = {key: test[key][indices] for key in input_keys}

user_vectors = user_model.predict(
    inputs, batch_size=128, verbose=0
)

user_vectors /= np.maximum(
    np.linalg.norm(user_vectors, axis=1, keepdims=True),
    1e-12
)

# 7. 分别执行四路召回
metrics = {
    name: [0, 0.0]
    for name in [
        "UserPreference",
        "ItemEmbedding",
        "GlobalTrending",
        "YouTubeDNN",
        "SnakeMerge",
        "WithoutPreference",
        "WithoutItemEmbedding",
        "WithoutTrending",
        "WithoutYouTubeDNN",
        "Weighted_A",
        "Weighted_B",
        "Weighted_C"
    ]
}

for j, i in enumerate(indices):
    uid = test_users[i]
    target = test_targets[i]
    history = histories.get(uid, [])

    # 用户偏好召回
    counts = Counter(
        g for mid in history
        for g in genre_map.get(mid, set())
    )

    preferred = {g for g, _ in counts.most_common(3)}

    preference = [
        mid for mid in preference_pool
        if preferred & genre_map.get(mid, set())
    ][:K]

    # 物品向量召回
    hist_encoded = test["hist_movie_id"][i]
    recent_idx = hist_encoded[hist_encoded != 0][-1] - 1

    item_scores = embeddings @ embeddings[recent_idx]
    item_scores[recent_idx] = -np.inf

    item_result = movie_ids[
        np.argsort(item_scores)[::-1][:K]
    ].tolist()

    # YouTubeDNN 召回
    user_scores = embeddings @ user_vectors[j]

    dnn_result = movie_ids[
        np.argsort(user_scores)[::-1][:K]
    ].tolist()

    # 四路融合
    merged = snake_merge(
        [preference, item_result, popular, dnn_result], K
    )

    results = [
        preference,
        item_result,
        popular,
        dnn_result,
        merged,

        # 去掉用户偏好
        snake_merge(
            [item_result, popular, dnn_result], K
        ),

        # 去掉物品向量
        snake_merge(
            [preference, popular, dnn_result], K
        ),

        # 去掉热门
        snake_merge(
            [preference, item_result, dnn_result], K
        ),

        # 去掉双塔
        snake_merge(
            [preference, item_result, popular], K
        ),

        weighted_merge(
            [preference, item_result, popular, dnn_result],
            [1, 4, 4, 1], K
        ),

        weighted_merge(
            [preference, item_result, popular, dnn_result],
            [1, 5, 3, 1], K
        ),

        weighted_merge(
            [preference, item_result, popular, dnn_result],
            [1, 5, 4, 0], K
        )
    ]

    # 8. 计算评估指标
    for name, candidates in zip(metrics, results):
        if target in candidates:
            rank = candidates.index(target) + 1
            metrics[name][0] += 1
            metrics[name][1] += 1 / math.log2(rank + 1)

# 9. 输出实验结果
print("\n===== Recall Evaluation =====")
print("Test users:", N_USERS)

for name, (hits, ndcg) in metrics.items():
    hr = hits / N_USERS
    print(
        f"{name:16s} "
        f"HR@{K}={hr:.4f} "
        f"Recall@{K}={hr:.4f} "
        f"NDCG@{K}={ndcg / N_USERS:.4f}"
    )
