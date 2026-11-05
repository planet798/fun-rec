import numpy as np

# 两种属性的 Embedding
embedding = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

# 假设模型学到的原始权重参数
scores = np.array([1.0, 2.0])

# Softmax 归一化
weights = np.exp(scores) / np.exp(scores).sum()

# 加权聚合
result = np.sum(
    embedding * weights[:, None],
    axis = 0
)

print("权重：", weights)
print("最终 Embedding:", result)