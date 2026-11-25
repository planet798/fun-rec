import os
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from offline.feature.preprocess_retrieval import (
    process_features,
    generate_train_eval_samples
)

# 1. 加载数据
raw = Path(os.environ["FUNREC_RAW_DATA_PATH"])

output_dir = Path(
    "../tmp/web_project/temporal_experiment"
)

ratings = pd.read_pickle(
    output_dir / "train_ratings.pkl"
)

users = pd.read_pickle(raw / "users.pkl")
movies = pd.read_pickle(raw / "movies.pkl")

# 2. 特征编码
merged, user_vocab, movie_vocab, _, _ = process_features(
    movies, ratings, users
)

# 3. 构建滑动窗口样本
samples = generate_train_eval_samples(
    merged,
    user_columns=[
        "gender", "age", "occupation", "zip_code"
    ],
    item_columns=[
        "movie_id", "genres", "isAdult", "startYear"
    ],
    max_hist_seq_len=10,
    max_feat_seq_len=10
)

# 4. 合并全部训练期样本
train_samples = {
    key: np.concatenate(
        [samples["train"][key], samples["test"][key]],
        axis=0
    )
    for key in samples["train"]
}

# 5. 保存词表和特征配置
vocab_dict = {**user_vocab, **movie_vocab}

feature_dict = {
    key: len(value) + 1
    for key, value in vocab_dict.items()
}

artifacts = {
    "train_samples.pkl": train_samples,
    "vocab_dict.pkl": vocab_dict,
    "feature_dict.pkl": feature_dict
}

for name, data in artifacts.items():
    with open(output_dir / name, "wb") as f:
        pickle.dump(data, f)

# 6. 检查样本维度
print("\n===== Training Samples =====")

for key, value in train_samples.items():
    print(key, value.shape)

print("训练样本构建完成")