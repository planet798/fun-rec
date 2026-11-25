
import os
import math
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from funrec.models.layers import DNNs, L2NormalizeLayer


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

load_dotenv(".env")

K = 100
root = Path("../tmp/web_project/temporal_experiment")
raw = Path(os.environ["FUNREC_RAW_DATA_PATH"])

# 1. 加载训练集和验证集
train = pd.read_pickle(root / "train_ratings.pkl")
val = pd.read_pickle(root / "val_ratings.pkl")
users = pd.read_pickle(raw / "users.pkl")
movies = pd.read_pickle(raw / "movies.pkl")

for df in [train, val, users]:
    df["user_id"] = df["user_id"].astype(str)

for df in [train, val, movies]:
    df["movie_id"] = df["movie_id"].astype(str)

train["rating"] = pd.to_numeric(train["rating"])

# 2. 加载新模型和词表
with open(root / "vocab_dict.pkl", "rb") as f:
    vocab = pickle.load(f)

encoders = {
    key: {
        str(v): i + 1
        for i, v in enumerate(values)
    }
    for key, values in vocab.items()
}

model = tf.keras.models.load_model(
    root / "saved_models/user_model",
    custom_objects={
        "DNNs": DNNs,
        "L2NormalizeLayer": L2NormalizeLayer
    },
    compile=False
)

embeddings = np.load(root / "item_embeddings.npy")
movie_ids = np.load(root / "movie_ids.npy").astype(str)

assert np.array_equal(
    movie_ids,
    np.asarray(vocab["movie_id"]).astype(str)
)

# 仅允许召回训练期出现过的电影
known_movies = set(train["movie_id"])

valid_mask = np.isin(
    movie_ids, list(known_movies)
)

# 3. 构造历史和验证答案
histories = train.sort_values("timestamp").groupby(
    "user_id"
)["movie_id"].agg(list).to_dict()

targets = val.groupby("user_id")[
    "movie_id"
].agg(set).to_dict()

user_map = users.set_index("user_id").to_dict("index")

genre_map = {
    row.movie_id: str(row.genres).split("|")
    for row in movies.itertuples()
}

# 4. 仅根据训练集统计电影热度
stats = train.groupby("movie_id")["rating"].agg(
    avg_rating="mean",
    rating_count="count"
).reset_index()

movies = movies.merge(stats, on="movie_id", how="left")

movies["startYear"] = pd.to_numeric(
    movies["startYear"], errors="coerce"
)

popular_pool = movies[
    (movies["avg_rating"] >= 5) &
    (movies["rating_count"] >= 50)
].sort_values(
    ["rating_count", "avg_rating"],
    ascending=False
)["movie_id"].tolist()

preference_pool = movies[
    movies["avg_rating"] >= 5
].sort_values(
    ["startYear", "avg_rating"],
    ascending=False
)["movie_id"].tolist()

# 5. 构建用户塔输入
def encode(key, value):
    return encoders[key].get(str(value), 0)

def pad(seq):
    seq = seq[-10:]
    return [0] * (10 - len(seq)) + seq

eval_users = list(targets)

input_keys = [
    "user_id", "gender", "age", "occupation",
    "zip_code", "hist_movie_id", "hist_genres"
]

inputs = {key: [] for key in input_keys}

for uid in eval_users:
    profile = user_map[uid]
    history = histories[uid]

    inputs["user_id"].append(encode("user_id", uid))

    for key in ["gender", "age", "occupation", "zip_code"]:
        inputs[key].append(encode(key, profile[key]))

    inputs["hist_movie_id"].append(
        pad([encode("movie_id", m) for m in history])
    )

    genres = [
        encode("genres", g)
        for m in history
        for g in genre_map.get(m, [])
    ]

    inputs["hist_genres"].append(pad(genres))

inputs = {
    key: np.asarray(value)
    for key, value in inputs.items()
}

user_vectors = model.predict(
    inputs, batch_size=128, verbose=0
)

user_vectors /= np.maximum(
    np.linalg.norm(user_vectors, axis=1, keepdims=True),
    1e-12
)

# 6. 计算评估指标
def evaluate(candidates, ground_truth):
    hits = [
        i for i, mid in enumerate(candidates)
        if mid in ground_truth
    ]

    hr = int(len(hits) > 0)
    recall = len(hits) / len(ground_truth)

    dcg = sum(
        1 / math.log2(i + 2)
        for i in hits
    )

    idcg = sum(
        1 / math.log2(i + 2)
        for i in range(min(K, len(ground_truth)))
    )

    return np.array([hr, recall, dcg / idcg])

# 7. 设置对比实验
weight_configs = {
    "Weighted_A": [1, 4, 4, 1],
    "Weighted_B": [1, 5, 3, 1],
    "Weighted_C": [1, 5, 4, 0],
    "Weighted_D": [4, 1, 4, 1],
    "Weighted_E": [3, 1, 6, 0],
    "Weighted_F": [4, 0, 6, 0],
    "Weighted_G": [5, 0, 5, 0]
}

names = [
    "UserPreference",
    "ItemEmbedding",
    "GlobalTrending",
    "YouTubeDNN",
    "SnakeMerge",
    *weight_configs.keys()
]

metrics = {
    name: np.zeros(3)
    for name in names
}

# 8. 执行四路召回
for i, uid in enumerate(eval_users):
    history = histories[uid]
    seen = set(history)
    truth = targets[uid]

    counts = Counter(
        g for mid in history
        for g in genre_map.get(mid, [])
    )

    preferred = {
        g for g, _ in counts.most_common(3)
    }

    preference = [
        mid for mid in preference_pool
        if mid not in seen
        and preferred.intersection(genre_map.get(mid, []))
    ][:K]

    popular = [
        mid for mid in popular_pool
        if mid not in seen
    ][:K]

    mask = valid_mask & ~np.isin(
        movie_ids, list(seen)
    )

    def topk(scores):
        scores = np.where(mask, scores, -np.inf)
        indices = np.argsort(scores)[::-1][:K]

        return [
            movie_ids[j]
            for j in indices
            if np.isfinite(scores[j])
        ]

    recent_idx = encode("movie_id", history[-1]) - 1

    item_result = topk(
        embeddings @ embeddings[recent_idx]
    )

    dnn_result = topk(
        embeddings @ user_vectors[i]
    )

    channels = [
        preference,
        item_result,
        popular,
        dnn_result
    ]

    results = channels + [
        snake_merge(channels, K)
    ]

    for weights in weight_configs.values():
        results.append(
            weighted_merge(channels, weights, K)
        )

    for name, candidates in zip(names, results):
        metrics[name] += evaluate(candidates, truth)

# 9. 输出验证集结果
print("\n===== Temporal Validation =====")
print("Validation users:", len(eval_users))

for name, values in metrics.items():
    hr, recall, ndcg = values / len(eval_users)

    print(
        f"{name:16s} "
        f"HR@100={hr:.4f} "
        f"Recall@100={recall:.4f} "
        f"NDCG@100={ndcg:.4f}"
    )
