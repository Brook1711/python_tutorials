import itertools

actions_tmp = [] # list of actions 2CM
actions_str_tmp = ''
for i in range(3):
    actions_str_tmp = actions_str_tmp + str(i)
        
for i in itertools.combinations(actions_str_tmp, 2):
    actions_tmp.append(''.join(i))
    actions_tmp.append(''.join(i[::-1]))