import random

total_num = 100
trg_ratio = 0.8
a = random.sample(range(total_num), int(total_num * trg_ratio))
print(a)