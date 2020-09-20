import system_model
import matplotlib.pyplot as plt
# 2020/9/20
my_model = system_model.System_model(2e6) #Create an object of the class, the total bandwidth is 2 MHz

print(1)
first_F  =   float(my_model.F) # the original F
best_F   =   float(my_model.F) # the original F
best_Bm  =   str(my_model.Bm)  # the original Bm[]
plot_F = [first_F] # record every F
num_actions = 0 #record the number of actions

for episode in range(10):
    while True:
    
        # debug vaneriables
        # watch_F = my_model.F
        
        #(1).Use current state and q_table to choose action(string) 
        action = my_model.choose_action(str(my_model.Bm))
       
        #(2).Use action(string) and present state to get the next state and reward
        state_last, reward, if_restart = my_model.step(action)

        #Record the highest F value and Bm[]
        #The self.parameters are global variables which is changinng during the whole process
        if my_model.F >= best_F:
            print("**************************arrangement found*******************************")
            print("     found new arrangemaet 'Bm'      :  " + str(my_model.Bm))
            print("under this arrangement, the F is     :  " + str(my_model.F))
            print("               the original F is     :  " + str(first_F))
            print("compare to the original F , F varies :  " + str(my_model.F - first_F))

            # debug vaneriables
            # watch_delta_F = best_F - first_F

            # iteration of F & best_Bm
            best_Bm = str(my_model.Bm)
            best_F = float(my_model.F)

        #(3).Use s, a, s_, r to update the q table  
        my_model.learn(state_last, action, reward, str(my_model.Bm))
        
        #This step must behind the update_q_table, 
        #because the state that is below the limit should also be recorded in the q table
        #if if_below_limit:
        #    my_model.Bm = state_last_list
        
        ###print(best_F)
        ###print(best_Bm)
        ###print("episode end")
        
        #Prepare for the plot
        plot_F.append(my_model.F)  
        num_actions = num_actions + 1
        

        if if_restart:
            my_model.restart()
            print("episode end")
            break

print("*****     " , best_F - first_F)
print(best_Bm)
print(num_actions)

"""Draw the convergence curve""" 
num_actions_list = []
for i in range (num_actions+1):
    num_actions_list.append(i)

fig = plt.figure()
x = num_actions_list
y = plot_F
plt.plot(x, y, 'g-')
plt.xlabel('step of actions')
plt.ylabel('F')
plt.title('convergence curve')
plt.show()