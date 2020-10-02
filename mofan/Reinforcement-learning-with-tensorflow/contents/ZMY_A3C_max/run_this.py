import brain
#2020/10/02
Mymodel = brain.System_A3C_Model(10000, 200)

###每个action前先算一下信道总增益h_total，因为要保证在一个action中h_total相等，防止在一个action中多次调用h函数计算