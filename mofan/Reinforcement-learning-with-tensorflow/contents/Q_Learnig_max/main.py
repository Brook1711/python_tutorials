import system_model
# 2020/9/13
my_model = system_model.System_model(2)

print(1)
first_F =  my_model.F
best_F = my_model.F
for episode in range(100):
    while True:
        # choose action from current state
        action = my_model.choose_action(str(my_model.Bm))
        state_last, reward, if_restart = my_model.step(action)
        if my_model.F > best_F:
            best_F = my_model.F
        
        my_model.learn(state_last, action, reward, str(my_model.Bm))
        if if_restart:
            my_model.restart()
            print(best_F)
            #print(my_model.F)
            print("episode end")
            break
        
print("*****     " , best_F - first_F)
