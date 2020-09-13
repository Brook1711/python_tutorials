import system_model
# 2020/9/13
my_model = system_model.System_model(2)

for episode in range(10):
    while True:
        # choose action from current state
        action = my_model.choose_action(my_model.Bm)
print(my_model.calculate_F())