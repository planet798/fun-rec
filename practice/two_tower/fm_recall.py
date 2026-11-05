import numpy as np
from user_tower import build_user_vector
from item_tower import build_item_vector

rng = np.random.default_rng(42)

user_embeddings = rng.normal(size = (4, 8))
item_embeddings = rng.normal(size =(5, 2, 8))

# 五部电影共享同一组线性权重
linear_weights = rng.normal(size = 8)

user_vector = build_user_vector(user_embeddings)
item_vectors = []

for item_emb in item_embeddings:
    item_vector = build_item_vector(
        item_emb, linear_weights
    )
    item_vectors.append(item_vector)

item_vectors = np.array(item_vectors)

# 计算用户与所有物品的内积
scores = np.dot(item_vectors, user_vector)

# 找出分数最高的三个物品索引
top3_indices = np.argsort(scores)[::-1][:3]

print("匹配分数:", scores)
print("Top-3物品索引:", top3_indices)