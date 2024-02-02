import queue

feature_que = set()
feature_que.add(1)
feature_que.add(2)
feature_que.add(3)
feature_list = list(feature_que)
if len(feature_list) > 2:
    feature_list.pop(0)
print(feature_list)
#feature_list = list(feature_que)
#feature_que.get()
#print(feature_que)