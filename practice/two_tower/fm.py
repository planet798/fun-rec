# 完成fm的两种二阶交互计算

import numpy as np

def fm_naive(x, V):
    score = 0
    # 原始双层循环计算二阶交互
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            score += np.dot(V[i], V[j])*x[i]*x[j]

    return score


def fm_fast(x, V):
    # 使用平方差公式计算二阶交互

    weighted_V = V * x[:, None]
    # 1. 先求和再平方，得到 A
    A = np.sum((np.sum(weighted_V, axis=0))**2)
    # 2. 先平方再求和，得到 B
    B = np.sum(weighted_V **2)

    return 0.5 * (A - B)

x = np.array([1., 0., 1., 1.])
V = np.random.randn(4, 8)

print(fm_naive(x, V))
print(fm_fast(x, V))
print(np.allclose(fm_naive(x, V), fm_fast(x, V)))