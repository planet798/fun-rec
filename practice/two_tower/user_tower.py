import numpy as np

# 模拟一个用户的4个特征，每个Embedding为8维
user_embeddings = np.random.randn(4, 8)

def build_user_vector(user_embeddings):
    # 1. 将4个Embedding逐元素相加
    embedding_sum = np.sum(user_embeddings, axis=0)

    # 2. 在向量前面拼接一个1
    user_vector = np.concatenate([
        np.array([1.0]),
        embedding_sum
    ])

    return user_vector

if __name__ == "__main__":
    # 原有测试代码
    print(build_user_vector(user_embeddings).shape)