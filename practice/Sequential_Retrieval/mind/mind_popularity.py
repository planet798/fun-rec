from collections import Counter
from mind_dataset import (
    DATA_PATH, load_sequences, build_samples
)

sequences = load_sequences(DATA_PATH)
_, test_samples = build_samples(sequences)

# 1：统计训练历史中每部电影出现的次数
pop_count = Counter()

for user_id, movies in sequences.items():
    if len(movies) < 3:
        continue

    # 只能使用 movies[:-1]，不能统计测试目标
    # 2：更新 pop_count
    pop_count.update(movies[:-1])

# 3：按照电影热度从高到低排序
ranked_movies = [
    movie_id
    for movie_id, count in pop_count.most_common()
]

def recommend_popularity(history, k=10):
    # 4：过滤用户已看过的电影
    history_set = set(history)

    filtered_movies = [
        movie_id
        for movie_id in ranked_movies
        if movie_id not in history_set
    ]
    # 5：返回前 k 部电影
    return filtered_movies[:k]

# 6：遍历 test_samples，计算 HR@10
total = len(test_samples)
hits = 0
warm_users = 0
cold_users = 0

for user_id, history, target in test_samples:

    if target not in ranked_movies:
        cold_users += 1
        continue

    warm_users += 1

    # 1：调用 recommend_top10(history)
    recommendations = recommend_popularity(history=history)

    # 3：判断 target_idx 是否在推荐结果中
    if target in recommendations:
        hits += 1

# 4：计算两个指标
hr10 = hits / total
warm_hr10 = hits / warm_users if warm_users > 0 else 0.0

print(f"HR@10: {hr10:.4f}")
print(f"Warm HR@10: {warm_hr10:.4f}")
print(f"冷启动测试用户数: {cold_users}")