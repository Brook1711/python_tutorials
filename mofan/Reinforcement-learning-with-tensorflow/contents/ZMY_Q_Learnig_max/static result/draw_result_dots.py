import matplotlib.pyplot as plt 
import ast

with open('dots.txt','r') as file_handle:   #
    txt_read = file_handle.read()
    txt_dic = ast.literal_eval(txt_read)
    txt_list = []
    x = []
    for i in range(len(txt_dic)):
        txt_list.append(float(txt_dic[str(i)]))
        x.append(i)

    plt.plot(x[650:1500], txt_list[650:1500], 'g-')
    plt.xlabel('step of actions')
    plt.ylabel('F')
    plt.title('convergence curve')
    plt.show()
    print(1)