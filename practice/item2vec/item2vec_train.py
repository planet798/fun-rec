from gensim.models import Word2Vec

import numpy as np

histories = [
    ["手机", "手机壳", "充电器"],
    ["手机", "手机壳", "耳机"],
    ["笔记本", "鼠标", "键盘"],
    ["笔记本", "鼠标", "显示器"],
    ["手机", "充电器", "耳机"],
    ["笔记本", "键盘", "显示器"],
]

model = Word2Vec(
    sentences=histories,
    vector_size=8,
    window=2,
    min_count=1,
    sg=1,
    workers=1,
    epochs=100,
    seed=42,
)

target_item = "手机"
target_vector = model.wv[target_item]

# 获取模型词表中的所有商品
items = model.wv.index_to_key
scores = []

for item in items:
    if item != "手机":
        vector = model.wv[item]

        similarity = np.dot(vector, target_vector) / (
            np.linalg.norm(vector) * np.linalg.norm(target_vector)
        )

        scores.append((item, similarity))


results = sorted(scores, key=lambda x : x[1], reverse=True)[:3]
print(results)