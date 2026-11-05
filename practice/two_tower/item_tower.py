import numpy as np

# 2个物品特征，每个Embedding为8维
item_embeddings = np.random.randn(2, 8)

# 模拟Dense(1)使用的权重
linear_weights = np.random.randn(8)

def build_item_vector(item_embeddings, linear_weights):

    # 1. 对物品Embedding逐元素求和
    embedding_sum = np.sum(item_embeddings, axis=0)

    # 2. 计算一阶线性项
    # 提示：embedding_sum与linear_weights内积
    linear_term = np.dot(embedding_sum, linear_weights)

    # 3. 使用之前学过的FM平方差公式计算二阶交互
    # 这里所有特征的x均为1
    A = np.sum(embedding_sum ** 2)
    B = np.sum(item_embeddings ** 2)
    fm_interaction = 0.5 * (A - B)

    # 4. 将一阶项和二阶交互相加，得到一个标量
    first_term = linear_term + fm_interaction

    # 5. 将first_term拼接到embedding_sum前面
    item_vector = np.concatenate(
        [np.array([first_term]),
        embedding_sum]
    )

    return item_vector

if __name__ == "__main__":
    # 原有测试代码
    item_vector = build_item_vector(
        item_embeddings,
        linear_weights
    )

    print(item_vector.shape)
    print(item_vector)