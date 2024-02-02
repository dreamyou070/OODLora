import os

file1 = 'normal_goodscore_mahalanobis_distances.txt'
file2 = 'normal_mahalanobis_distances.txt'
file3 = 'anormal_mahalanobis_distances.txt'

with open(file1, 'r') as f:
    lines1 = f.readlines()[0]
scores = lines1.split(',')
score_list = []
for s in scores :
    score = s.strip()
    try :
        score_list.append(float(score))
    except :
        pass
sample_num = len(score_list)
mean_value = sum(score_list) / sample_num
max_value = max(score_list)
min_value = min(score_list)
print(mean_value, max_value, min_value)

with open(file3, 'r') as f:
    lines3 = f.readlines()[0]
scores = lines3.split(',')
score_list = []
for s in scores :
    score = s.strip()
    try :
        score_list.append(float(score))
    except :
        pass
sample_num = len(score_list)
mean_value = sum(score_list) / sample_num
max_value = max(score_list)
min_value = min(score_list)
print(mean_value, max_value, min_value)