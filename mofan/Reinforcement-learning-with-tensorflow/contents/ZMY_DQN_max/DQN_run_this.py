import DQN_brain
#2020/10/07
import matplotlib.pyplot as plt
import numpy as np

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

step = 0

for episode in range(10):
    while True:
        for m in range(Mymodel.M):
            # 每个action前先算一下信道总增益h_total，
            # 因为要保证在一个action中h_total相等，防止在一个action中多次调用h函数计算
            # Mymodel.Hm.append(Mymodel.calculate_fading(Mymodel.Dm[m])) # wrong !!! 
            Mymodel.Hm[m] = Mymodel.calculate_fading(Mymodel.Dm[m])
        
        state = []
        for m in range(Mymodel.M):
            state.append(Mymodel.Bm[m])
        for m in range(Mymodel.M):
            state.append(Mymodel.Pm[m])
        
        state_array = np.array(state)
        # (1).Use current state and evaluate net to choose action(string) 
        action, action_num = Mymodel.choose_action(state_array)
       
        # (2).Use action(string) and present state to get the next state and reward
        state_last_array, reward, if_restart = Mymodel.step(action)

        # (3).Record the transition in repay memory
        Mymodel.store_transition(state_last_array, action_num, reward, state_array)

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

        #(3).Use s, a, s_, r to update the q table  
        ##Mymodel.learn()
        if (step > 200) and (step % 5 == 0):
            Mymodel.learn()
        
        #This step must behind the update_q_table, 
        #because the state that is below the limit should also be recorded in the q table
        #if if_below_limit:
        #    my_model.Bm = state_last_list
        
        ###print(best_F)
        ###print(best_Bm)
        ###print("episode end")
        
        #Prepare for the plot
        plot_umk.append(Mymodel.umk)  
        num_actions = num_actions + 1
        
        if if_restart:
            Mymodel.restart()
            print("episode end")
            break

        step = step + 1

print("*****     " , best_umk - first_umk)
print(best_Bm)
print(best_Pm)
print(num_actions)

"""Draw the convergence curve""" 
num_actions_list = []
for i in range (num_actions + 1):
    num_actions_list.append(i)

fig = plt.figure()
x = num_actions_list
y = plot_umk
plt.plot(x, y, 'g-')
plt.xlabel('step of actions')
plt.ylabel('umk')
plt.title('convergence curve')
plt.show()