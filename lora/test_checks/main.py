import os

file1 = 'normal_goodscore_mahalanobis_distances.txt'
file2 = 'anormal_mahalanobis_distances.txt'

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
print(max(score_list))

with open(file2, 'r') as f:
    lines2 = f.readlines()[0]
scores = lines2.split(',')
score_list = []
for s in scores :
    score = s.strip()
    try :
        score_list.append(float(score))
    except :
        pass
print(max(score_list))
