# 全局上下文正样本生成

session = ["A", "B", "C", "D"]
booked_item = "D"

global_pairs = []

for item in session:
    if item != booked_item:
        global_pairs.append((item, booked_item))

print(global_pairs)