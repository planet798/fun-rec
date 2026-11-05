import numpy as np

user_vector = np.array([3.0, 4.0])
item_vector = np.array([4.0, 3.0])

def l2_normalize(vector):
    # 1. 计算向量L2范数
    norm = np.linalg.norm(vector)

    # 2. 向量归一化
    normalized_vector = vector / norm

    return normalized_vector

# 3. 分别归一化用户、物品向量
user_embedding = l2_normalize(user_vector)
item_embedding = l2_normalize(item_vector)

# 4. 计算两个归一化向量的内积
score = np.dot(user_embedding, item_embedding)

print("用户向量:", user_embedding)
print("物品向量:", item_embedding)
print("余弦相似度:", score)