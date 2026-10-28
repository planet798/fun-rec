words = ["我", "喜欢", "吃", "苹果"]

windows = 1
pairs = []

for i, center in enumerate(words):
    for j in range(max(0, i - windows), min(len(words), i + windows + 1)):
        if i != j:
            pairs.append((center, words[j]))

print(pairs)        