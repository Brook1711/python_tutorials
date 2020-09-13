import itertools

action_tmp = []
for i in itertools.combinations(range(3), 2):
    i=list(i)
    action_tmp.append(i)
    i.reverse()
    action_tmp.append(i)
print(action_tmp)