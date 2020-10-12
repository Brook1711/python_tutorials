import DQN_brain
#2020/10/07
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spy 
from queue import Queue

Mymodel = DQN_brain.System_DQN_Model(10000, 200)
print(1)
first_umk = float(Mymodel.umk) # the original umk
best_umk = float(Mymodel.umk) # the original umk
best_Bm = str(Mymodel.Bm) # the original Bm[]
best_Pm = str(Mymodel.Pm) # the original Pm[]  
plot_umk = [first_umk] # record every umk
num_actions = 0 #record the number of actions

#first_umk = Mymodel.umk # the original umk
#best_umk = Mymodel.umk # the original umk
#best_Bm = Mymodel.Bm # the original Bm[]
#best_Pm = Mymodel.Pm # the original Pm[]  
#plot_umk = [first_umk] # record every umk
#num_actions = 0 #record the number of actions
#max_step = 100

#总的执行10000步，每个episode执行100个action
    
state = []
for m in range(Mymodel.M):
    state.append(Mymodel.Bm[m])
for m in range(Mymodel.M):
    state.append(Mymodel.Pm[m])

state_array = np.array(state) # record the Bm and Pm as an array

for episode in range(200):  #org 800
    while True:
        
        #step = 0 #record the number of actions

        for m in range(Mymodel.M):
            # 每个action前先算一下信道总增益h_total，
            # 因为要保证在一个action中h_total相等，防止在一个action中多次调用h函数计算
            # Mymodel.Hm.append(Mymodel.calculate_fading(Mymodel.Dm[m])) # wrong !!! 
            Mymodel.Hm[m] = Mymodel.calculate_fading(Mymodel.Dm[m])
            # Mymodel.Hm[m] = 3.717e-13

        # (1).Use current state and evaluate net to choose action(string) 
        action, action_num = Mymodel.choose_action(state_array)
       
        # (2).Use action(string) and present state to get the next state and reward
        state_last_array, reward, if_restart = Mymodel.step(action)

        for m in range(Mymodel.M):
            state_array[m] = Mymodel.Bm[m]
        for m in range(Mymodel.M):
            state_array[m + 3] = Mymodel.Pm[m]  

        # (3).Record the transition in repay memory
        Mymodel.store_transition(state_last_array, action_num, reward, state_array)

        if (num_actions > 200) and (num_actions % 5 == 0):
            Mymodel.learn()

        # Record the highest umk value and Bm[], Pm[]
        # The self.parameters are global variables which is changing during the whole process
        if Mymodel.umk >= best_umk:
            print("**************************arrangement found*******************************")
            print("       found new arrangemant 'Bm'       :  " + str(Mymodel.Bm))
            print("       found new arrangemant 'Pm'       :  " + str(Mymodel.Pm))
            print("   under this arrangement, the umk is   :  " + str(Mymodel.umk))
            print("           the original umk is          :  " + str(first_umk))
            print("compare to the original umk, umk varies :  " + str(Mymodel.umk - first_umk))

            # debug vaneriables
            # watch_delta_F = best_F - first_F

            # iteration of F & best_Bm
            best_Bm = str(Mymodel.Bm)
            best_Pm = str(Mymodel.Pm)
            best_umk = float(Mymodel.umk)
        
        #Prepare for the plot
        plot_umk.append(Mymodel.umk)  
        #step = step + 1 
        num_actions = num_actions + 1

        #if num_actions == 3000:
        #    Mymodel.epsilon = 0.95
        
        #if num_actions == 4000:
        #    Mymodel.epsilon = 1
        
        if if_restart or num_actions % 100 == 0:
            Mymodel.restart()
            print("episode end")
            break

        if num_actions>500:
            Mymodel.epsilon = 0.85
        if num_actions>1000:
            Mymodel.epsilon = 0.9
        if num_actions>2000:
            Mymodel.epsilon = 0.95
        if num_actions>3000:
            Mymodel.epsilon = 1

#for episode in range(100):
#    while True:
#        
#        #step = 0 #record the number of actions
#
#        for m in range(Mymodel.M):
#            # 每个action前先算一下信道总增益h_total，
#            # 因为要保证在一个action中h_total相等，防止在一个action中多次调用h函数计算
#            # Mymodel.Hm.append(Mymodel.calculate_fading(Mymodel.Dm[m])) # wrong !!! 
#            Mymodel.Hm[m] = Mymodel.calculate_fading(Mymodel.Dm[m])
#            # Mymodel.Hm[m] = 3.717e-13
#
#        # (1).Use current state and evaluate net to choose action(string) 
#        action, action_num = Mymodel.choose_action(state_array)
#       
#        # (2).Use action(string) and present state to get the next state and reward
#        state_last_array, reward, if_restart = Mymodel.step(action)
#
#        for m in range(Mymodel.M):
#            state_array[m] = Mymodel.Bm[m]
#        for m in range(Mymodel.M):
#            state_array[m + 3] = Mymodel.Pm[m]  
#
#        # (3).Record the transition in repay memory
#        Mymodel.store_transition(state_last_array, action_num, reward, state_array)
#
#        if (num_actions > 200) and (num_actions % 5 == 0):
#            Mymodel.learn()
#
#        # Record the highest umk value and Bm[], Pm[]
#        # The self.parameters are global variables which is changing during the whole process
#        if Mymodel.umk >= best_umk:
#            print("**************************arrangement found*******************************")
#            print("       found new arrangemant 'Bm'       :  " + str(Mymodel.Bm))
#            print("       found new arrangemant 'Pm'       :  " + str(Mymodel.Pm))
#            print("   under this arrangement, the umk is   :  " + str(Mymodel.umk))
#            print("           the original umk is          :  " + str(first_umk))
#            print("compare to the original umk, umk varies :  " + str(Mymodel.umk - first_umk))
#
#            # debug vaneriables
#            # watch_delta_F = best_F - first_F
#
#            # iteration of F & best_Bm
#            best_Bm = str(Mymodel.Bm)
#            best_Pm = str(Mymodel.Pm)
#            best_umk = float(Mymodel.umk)
#       
#        #Prepare for the plot
#        plot_umk.append(Mymodel.umk)  
#        #step = step + 1 
#        num_actions = num_actions + 1
#
#        #if num_actions == 3000:
#        #    Mymodel.epsilon = 0.95
#        
#        #if num_actions == 4000:
#        #    Mymodel.epsilon = 1
#        
#        if if_restart:
#            Mymodel.restart()
#            print("episode end")
#            break

print("*****     " , best_umk - first_umk)
print(best_Bm)
print(best_Pm)
print(best_umk)
print(num_actions)
print(Mymodel.epsilon)

"""Draw the convergence curve""" 

# smooth the figure
smooth_dot_y = [] # record every average y
start = 0 # record the window's start, slide it in one step in every loop 
win = 100 # window size

for i in range(num_actions+1-win): # loop number is total actions number plus one minus window size
    temp_store_y = Queue(maxsize = win) # defne the window as a queue
    for count in range(win): # record the y in the window
        temp_store_y.put( plot_umk[count + start] )
    temp_sum = 0
    for count in range(win): # sum up all the y in the window
        temp_sum = temp_sum + temp_store_y.queue[count]
    smooth_dot_y.append(temp_sum / win)
    temp_store_y.get() # output the "first in" number
    start = start + 1 #slide the window in one step


#smooth = spy.savgol_filter(x, 553, 3)
#plt.plot(smooth,y,'g-')
#num_actions_list = []
#for i in range (num_actions + 1):
#    num_actions_list.append(i)

#fig = plt.figure()
#x = num_actions_list
#y = plot_umk

num_actions_list = []
for i in range (num_actions+1-win):
    num_actions_list.append(i)

x = num_actions_list
y = smooth_dot_y

fig = plt.figure()
plt.plot(x, y, 'g-')
plt.xlabel('step of actions')
plt.ylabel('umk')
plt.title('convergence curve')
plt.show()