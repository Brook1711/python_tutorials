import DQN_brain
#2020/10/28
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spy 
#from queue import Queue
import codecs

"""Model 0:"""
Mymodel_0 = DQN_brain.System_DQN_Model(10000, 200)
print("number 0 ")
first_umk_0 = float(Mymodel_0.umk) # the original umk
best_umk_0 = float(Mymodel_0.umk) # the original umk
best_Bm_0 = str(Mymodel_0.Bm) # the original Bm[]
best_Pm_0 = str(Mymodel_0.Pm) # the original Pm[]  
plot_umk_0 = [first_umk_0] # record every umk
num_actions_0 = 0 #record the number of actions
    
state_0 = []
for m in range(Mymodel_0.M):
    state_0.append(Mymodel_0.Bm[m])
for m in range(Mymodel_0.M):
    state_0.append(Mymodel_0.Pm[m])

state_array_0 = np.array(state_0) # record the Bm and Pm as an array

for episode in range(200):  #org 800
    while True:

        for m in range(Mymodel_0.M):
            # 每个action前先算一下信道总增益h_total，
            # 因为要保证在一个action中h_total相等，防止在一个action中多次调用h函数计算
            # Mymodel.Hm.append(Mymodel.calculate_fading(Mymodel.Dm[m])) # wrong !!! 
            Mymodel_0.Hm[m] = Mymodel_0.calculate_fading(Mymodel_0.Dm[m])
            # Mymodel.Hm[m] = 3.717e-13

        # (1).Use current state and evaluate net to choose action(string) 
        action_0, action_num_0 = Mymodel_0.choose_action(state_array_0)
       
        # (2).Use action(string) and present state to get the next state and reward
        state_last_array_0, reward_0, if_restart_0 = Mymodel_0.step(action_0)

        for m in range(Mymodel_0.M):
            state_array_0[m] = Mymodel_0.Bm[m]
        for m in range(Mymodel_0.M):
            state_array_0[m + 3] = Mymodel_0.Pm[m]  

        # (3).Record the transition in repay memory
        Mymodel_0.store_transition(state_last_array_0, action_num_0, reward_0, state_array_0)

        if (num_actions_0 > 200) and (num_actions_0 % 5 == 0):
            Mymodel_0.learn()

        # Record the highest umk value and Bm[], Pm[]
        # The self.parameters are global variables which is changing during the whole process
        if Mymodel_0.umk >= best_umk_0:
            print("**************************arrangement found*******************************")
            print("       found new arrangemant 'Bm'       :  " + str(Mymodel_0.Bm))
            print("       found new arrangemant 'Pm'       :  " + str(Mymodel_0.Pm))
            print("   under this arrangement, the umk is   :  " + str(Mymodel_0.umk))
            print("           the original umk is          :  " + str(first_umk_0))
            print("compare to the original umk, umk varies :  " + str(Mymodel_0.umk - first_umk_0))

            # iteration of F & best_Bm
            best_Bm_0 = str(Mymodel_0.Bm)
            best_Pm_0 = str(Mymodel_0.Pm)
            best_umk_0 = float(Mymodel_0.umk)
        
        #Prepare for the plot
        plot_umk_0.append(Mymodel_0.umk)  
        #step = step + 1 
        num_actions_0 = num_actions_0 + 1
        
        if if_restart_0 or num_actions_0 % 100 == 0:
            Mymodel_0.restart()
            print("episode end")
            break

        if num_actions_0>500:
            Mymodel_0.epsilon = 0.85
        if num_actions_0>1000:
            Mymodel_0.epsilon = 0.9
        if num_actions_0>2000:
            Mymodel_0.epsilon = 0.95
        if num_actions_0>3000:
            Mymodel_0.epsilon = 1

print("*****     " , best_umk_0 - first_umk_0)
print(best_Bm_0)
print(best_Pm_0)
print(best_umk_0)
print(num_actions_0)

# record the data 
num_actions_list_0 = []
sum_y_0 = 0
aver_y_0 = []
for i in range (num_actions_0 + 1):
    num_actions_list_0.append(i)
    sum_y_0 = sum_y_0 + plot_umk_0[i]
    aver_y_0.append(sum_y_0/(i+1)) 

x_0 = num_actions_list_0
y_0 = aver_y_0

points_tulpe_0 = list(zip(x_0, y_0))
#print(points_tulpe_0)
f = codecs.open('write_B_200Hz_P_10000mW.txt','w')       # w 表示写
f.writelines(str(points_tulpe_0)+'\n')          # 将元组写入文件

"""Model 1:"""
Mymodel_1 = DQN_brain.System_DQN_Model(10000, 400)
print("number 1 ")
first_umk_1 = float(Mymodel_1.umk) # the original umk
best_umk_1 = float(Mymodel_1.umk) # the original umk
best_Bm_1 = str(Mymodel_1.Bm) # the original Bm[]
best_Pm_1 = str(Mymodel_1.Pm) # the original Pm[]  
plot_umk_1 = [first_umk_1] # record every umk
num_actions_1 = 0 #record the number of actions
    
state_1 = []
for m in range(Mymodel_1.M):
    state_1.append(Mymodel_1.Bm[m])
for m in range(Mymodel_1.M):
    state_1.append(Mymodel_1.Pm[m])

state_array_1 = np.array(state_1) # record the Bm and Pm as an array

for episode in range(200):  #org 800
    while True:

        for m in range(Mymodel_1.M):
            # 每个action前先算一下信道总增益h_total，
            # 因为要保证在一个action中h_total相等，防止在一个action中多次调用h函数计算
            # Mymodel.Hm.append(Mymodel.calculate_fading(Mymodel.Dm[m])) # wrong !!! 
            Mymodel_1.Hm[m] = Mymodel_1.calculate_fading(Mymodel_1.Dm[m])
            # Mymodel.Hm[m] = 3.717e-13

        # (1).Use current state and evaluate net to choose action(string) 
        action_1, action_num_1 = Mymodel_1.choose_action(state_array_1)
       
        # (2).Use action(string) and present state to get the next state and reward
        state_last_array_1, reward_1, if_restart_1 = Mymodel_1.step(action_1)

        for m in range(Mymodel_1.M):
            state_array_1[m] = Mymodel_1.Bm[m]
        for m in range(Mymodel_1.M):
            state_array_1[m + 3] = Mymodel_1.Pm[m]  

        # (3).Record the transition in repay memory
        Mymodel_1.store_transition(state_last_array_1, action_num_1, reward_1, state_array_1)

        if (num_actions_1 > 200) and (num_actions_1 % 5 == 0):
            Mymodel_1.learn()

        # Record the highest umk value and Bm[], Pm[]
        # The self.parameters are global variables which is changing during the whole process
        if Mymodel_1.umk >= best_umk_1:
            print("**************************arrangement found*******************************")
            print("       found new arrangemant 'Bm'       :  " + str(Mymodel_1.Bm))
            print("       found new arrangemant 'Pm'       :  " + str(Mymodel_1.Pm))
            print("   under this arrangement, the umk is   :  " + str(Mymodel_1.umk))
            print("           the original umk is          :  " + str(first_umk_1))
            print("compare to the original umk, umk varies :  " + str(Mymodel_1.umk - first_umk_1))

            # iteration of F & best_Bm
            best_Bm_1 = str(Mymodel_1.Bm)
            best_Pm_1 = str(Mymodel_1.Pm)
            best_umk_1 = float(Mymodel_1.umk)
        
        #Prepare for the plot
        plot_umk_1.append(Mymodel_1.umk)  
        #step = step + 1 
        num_actions_1 = num_actions_1 + 1
        
        if if_restart_1 or num_actions_1 % 100 == 0:
            Mymodel_1.restart()
            print("episode end")
            break

        if num_actions_1>500:
            Mymodel_1.epsilon = 0.85
        if num_actions_1>1000:
            Mymodel_1.epsilon = 0.9
        if num_actions_1>2000:
            Mymodel_1.epsilon = 0.95
        if num_actions_1>3000:
            Mymodel_1.epsilon = 1

print("*****     " , best_umk_1 - first_umk_1)
print(best_Bm_1)
print(best_Pm_1)
print(best_umk_1)
print(num_actions_1)

# record the data 
num_actions_list_1 = []
sum_y_1 = 0
aver_y_1 = []
for i in range (num_actions_1 + 1):
    num_actions_list_1.append(i)
    sum_y_1 = sum_y_1 + plot_umk_1[i]
    aver_y_1.append(sum_y_1/(i+1)) 

x_1 = num_actions_list_1
y_1 = aver_y_1

points_tulpe_1 = list(zip(x_1, y_1))
#print(points_tulpe_1)
f = codecs.open('write_B_400Hz_P_10000mW.txt','w')       # w 表示写
f.writelines(str(points_tulpe_1)+'\n')          # 将元组写入文件

"""Model 2:"""
Mymodel_2 = DQN_brain.System_DQN_Model(20000, 200)
print("number 2 ")
first_umk_2 = float(Mymodel_2.umk) # the original umk
best_umk_2 = float(Mymodel_2.umk) # the original umk
best_Bm_2 = str(Mymodel_2.Bm) # the original Bm[]
best_Pm_2 = str(Mymodel_2.Pm) # the original Pm[]  
plot_umk_2 = [first_umk_2] # record every umk
num_actions_2 = 0 #record the number of actions
    
state_2 = []
for m in range(Mymodel_2.M):
    state_2.append(Mymodel_2.Bm[m])
for m in range(Mymodel_2.M):
    state_2.append(Mymodel_2.Pm[m])

state_array_2 = np.array(state_2) # record the Bm and Pm as an array

for episode in range(200):  #org 800
    while True:

        for m in range(Mymodel_2.M):
            # 每个action前先算一下信道总增益h_total，
            # 因为要保证在一个action中h_total相等，防止在一个action中多次调用h函数计算
            # Mymodel.Hm.append(Mymodel.calculate_fading(Mymodel.Dm[m])) # wrong !!! 
            Mymodel_2.Hm[m] = Mymodel_2.calculate_fading(Mymodel_2.Dm[m])
            # Mymodel.Hm[m] = 3.717e-13

        # (1).Use current state and evaluate net to choose action(string) 
        action_2, action_num_2 = Mymodel_2.choose_action(state_array_2)
       
        # (2).Use action(string) and present state to get the next state and reward
        state_last_array_2, reward_2, if_restart_2 = Mymodel_2.step(action_2)

        for m in range(Mymodel_2.M):
            state_array_2[m] = Mymodel_2.Bm[m]
        for m in range(Mymodel_2.M):
            state_array_2[m + 3] = Mymodel_2.Pm[m]  

        # (3).Record the transition in repay memory
        Mymodel_2.store_transition(state_last_array_2, action_num_2, reward_2, state_array_2)

        if (num_actions_2 > 200) and (num_actions_2 % 5 == 0):
            Mymodel_2.learn()

        # Record the highest umk value and Bm[], Pm[]
        # The self.parameters are global variables which is changing during the whole process
        if Mymodel_2.umk >= best_umk_2:
            print("**************************arrangement found*******************************")
            print("       found new arrangemant 'Bm'       :  " + str(Mymodel_2.Bm))
            print("       found new arrangemant 'Pm'       :  " + str(Mymodel_2.Pm))
            print("   under this arrangement, the umk is   :  " + str(Mymodel_2.umk))
            print("           the original umk is          :  " + str(first_umk_2))
            print("compare to the original umk, umk varies :  " + str(Mymodel_2.umk - first_umk_2))

            # iteration of F & best_Bm
            best_Bm_2 = str(Mymodel_2.Bm)
            best_Pm_2 = str(Mymodel_2.Pm)
            best_umk_2 = float(Mymodel_2.umk)
        
        #Prepare for the plot
        plot_umk_2.append(Mymodel_2.umk)  
        #step = step + 1 
        num_actions_2 = num_actions_2 + 1
        
        if if_restart_2 or num_actions_2 % 100 == 0:
            Mymodel_2.restart()
            print("episode end")
            break

        if num_actions_2>500:
            Mymodel_2.epsilon = 0.85
        if num_actions_2>1000:
            Mymodel_2.epsilon = 0.9
        if num_actions_2>2000:
            Mymodel_2.epsilon = 0.95
        if num_actions_2>3000:
            Mymodel_2.epsilon = 1

print("*****     " , best_umk_2 - first_umk_2)
print(best_Bm_2)
print(best_Pm_2)
print(best_umk_2)
print(num_actions_2)

# record the data 
num_actions_list_2 = []
sum_y_2 = 0
aver_y_2 = []
for i in range (num_actions_2 + 1):
    num_actions_list_2.append(i)
    sum_y_2 = sum_y_2 + plot_umk_2[i]
    aver_y_2.append(sum_y_2/(i+1)) 

x_2 = num_actions_list_2
y_2 = aver_y_2

points_tulpe_2 = list(zip(x_2, y_2))
#print(points_tulpe_2)
f = codecs.open('write_B_200Hz_P_20000mW.txt','w')       # w 表示写
f.writelines(str(points_tulpe_2)+'\n')          # 将元组写入文件

"""Model 3:"""
Mymodel_3 = DQN_brain.System_DQN_Model(20000, 400)
print("number 3 ")
first_umk_3 = float(Mymodel_3.umk) # the original umk
best_umk_3 = float(Mymodel_3.umk) # the original umk
best_Bm_3 = str(Mymodel_3.Bm) # the original Bm[]
best_Pm_3 = str(Mymodel_3.Pm) # the original Pm[]  
plot_umk_3 = [first_umk_3] # record every umk
num_actions_3 = 0 #record the number of actions
    
state_3 = []
for m in range(Mymodel_3.M):
    state_3.append(Mymodel_3.Bm[m])
for m in range(Mymodel_3.M):
    state_3.append(Mymodel_3.Pm[m])

state_array_3 = np.array(state_3) # record the Bm and Pm as an array

for episode in range(200):  #org 800
    while True:

        for m in range(Mymodel_3.M):
            # 每个action前先算一下信道总增益h_total，
            # 因为要保证在一个action中h_total相等，防止在一个action中多次调用h函数计算
            # Mymodel.Hm.append(Mymodel.calculate_fading(Mymodel.Dm[m])) # wrong !!! 
            Mymodel_3.Hm[m] = Mymodel_3.calculate_fading(Mymodel_3.Dm[m])
            # Mymodel.Hm[m] = 3.717e-13

        # (1).Use current state and evaluate net to choose action(string) 
        action_3, action_num_3 = Mymodel_3.choose_action(state_array_3)
       
        # (2).Use action(string) and present state to get the next state and reward
        state_last_array_3, reward_3, if_restart_3 = Mymodel_3.step(action_3)

        for m in range(Mymodel_3.M):
            state_array_3[m] = Mymodel_3.Bm[m]
        for m in range(Mymodel_3.M):
            state_array_3[m + 3] = Mymodel_3.Pm[m]  

        # (3).Record the transition in repay memory
        Mymodel_3.store_transition(state_last_array_3, action_num_3, reward_3, state_array_3)

        if (num_actions_3 > 200) and (num_actions_3 % 5 == 0):
            Mymodel_3.learn()

        # Record the highest umk value and Bm[], Pm[]
        # The self.parameters are global variables which is changing during the whole process
        if Mymodel_3.umk >= best_umk_3:
            print("**************************arrangement found*******************************")
            print("       found new arrangemant 'Bm'       :  " + str(Mymodel_3.Bm))
            print("       found new arrangemant 'Pm'       :  " + str(Mymodel_3.Pm))
            print("   under this arrangement, the umk is   :  " + str(Mymodel_3.umk))
            print("           the original umk is          :  " + str(first_umk_3))
            print("compare to the original umk, umk varies :  " + str(Mymodel_3.umk - first_umk_3))

            # iteration of F & best_Bm
            best_Bm_3 = str(Mymodel_3.Bm)
            best_Pm_3 = str(Mymodel_3.Pm)
            best_umk_3 = float(Mymodel_3.umk)
        
        #Prepare for the plot
        plot_umk_3.append(Mymodel_3.umk)  
        #step = step + 1 
        num_actions_3 = num_actions_3 + 1
        
        if if_restart_3 or num_actions_3 % 100 == 0:
            Mymodel_3.restart()
            print("episode end")
            break

        if num_actions_3>500:
            Mymodel_3.epsilon = 0.85
        if num_actions_3>1000:
            Mymodel_3.epsilon = 0.9
        if num_actions_3>2000:
            Mymodel_3.epsilon = 0.95
        if num_actions_3>3000:
            Mymodel_3.epsilon = 1

print("*****     " , best_umk_3 - first_umk_3)
print(best_Bm_3)
print(best_Pm_3)
print(best_umk_3)
print(num_actions_3)

# record the data 
num_actions_list_3 = []
sum_y_3 = 0
aver_y_3 = []
for i in range (num_actions_3 + 1):
    num_actions_list_3.append(i)
    sum_y_3 = sum_y_3 + plot_umk_3[i]
    aver_y_3.append(sum_y_3/(i+1)) 

x_3 = num_actions_list_3
y_3 = aver_y_3

points_tulpe_3 = list(zip(x_3, y_3))
#print(points_tulpe_3)
f = codecs.open('write_B_400Hz_P_20000mW.txt','w')       # w 表示写
f.writelines(str(points_tulpe_3)+'\n')          # 将元组写入文件

fig = plt.figure()
plt.title('convergence curve')
plt.plot(x_0, y_0, color = 'green', label = 'B_total=200Hz  P_total=10000mW')
plt.plot(x_1, y_1, color = 'red', label = 'B_total=400Hz  P_total=10000mW')
plt.plot(x_2, y_2, color = 'blue', label = 'B_total=200Hz  P_total=20000mW')
plt.plot(x_3, y_3, color = 'yellow', label = 'B_total=400Hz  P_total=20000mW')
plt.legend() # show the four legends
plt.xlabel('step of actions')
plt.ylabel('umk')
plt.show()