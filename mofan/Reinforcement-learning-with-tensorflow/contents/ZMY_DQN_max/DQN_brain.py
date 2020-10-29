"""
Initialize the system model and build the A3C model
"""
import numpy as np
import pandas as pd
import random 
import math 
import itertools
from scipy.stats import norm

import tensorflow.compat.v1 as tf  #use tensorflow 1.0
tf.disable_v2_behavior()   #forbidden tensorflow 2.0
np.random.seed(1) # first stack of seeds(every stack have their own random seed)同一堆种子每次生成的随机数相同
tf.set_random_seed(1) # set every stack have the same seeds 不同Session中的random函数表现出相对协同的特征

class System_DQN_Model:
     # total power(mw) and bandwidth(Hz) must be changed in run_this file
    def __init__(self, P_total = 10000, B_total = 200) : 
        tf.reset_default_graph() # prepare to build 4 objects
        self.M = 3 # number of vehicles
        self.K = 3 # number of fames of every vehicles

        self.B_total = B_total
        self.P_total = P_total

        self.F = [[0.9, 0.9, 1], [0.5, 0.4, 0.5], [0.1, 0.2, 0]] # importance of each fames, range in[0,1] 
        #self.F = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

        Dm = [] # distance between each vehicles and edge server, range in [0,0.5km] (server cover area) 
        for m in range(self.M):
            Dm.append(0.3)
        self.Dm = Dm

        Pm = [] # power of each vehicles, it's initialized to be eventually distributed
        for m in range(self.M):
            Pm.append(self.P_total / self.M)
        #self.Pm = Pm
        self.Pm = [1/8* P_total, 3/8 *P_total, 1/2 * P_total]
        
        Bm = [] # bandwidth of each vehicles, it's initialized to be eventually distributed
        for m in range(self.M):
            Bm.append(self.B_total / self.M)
        #self.Bm = Bm
        self.Bm = [1/8 * B_total, 3/8 * B_total, 1/2 * B_total]
        
        self.N0 = math.pow(10, -174 / 10.0)  # noise power density(sigma_square) mW/Hz

        self.B_min = self.B_total / 10.0 # lower boundary of Bm
        self.P_min = self.P_total / 100.0 # lower boundary of Pm 

        self.deltaB = self.B_total / 100.0 # the mini change of Bm
        self.deltaP = self.P_total / 100.0 # the mini change of Pm
        
        Hm = [] # total loss and fading of each vehicles
        for m in range(self.M):
            Hm.append(self.calculate_fading(self.Dm[m]))
        self.Hm = Hm

        # model parameters
        # use in the SSIM
        # low
        self.a = -4.247e-10
        self.b = 5.1
        self.c = 0.9521

        # high
        #self.a = -1.627e-8
        #self.b = 4.243
        #self.c = 0.9513

        # use in the QP
        self.alpha = 45.96
        self.beta = -8.648e-5

        # quantitative order
        self.M_quant = 2

        # modulation order e.g.:QPSK
        self.M_modul = 4

        # use in error rate
        self.A = 2 * (1 - 1 / math.pow(self.M_modul, 0.5)) / math.log2(math.pow(self.M_modul, 0.5))
        self.B = 3 * math.log2(math.pow(self.M_modul, 0.5)) / (self.M_modul - 1)

        # DQN parameters
        self.n_actions = 6*6 # action numbers (6 Pm change + 6 Bm change)
        self.n_features = 2*3 # one state has sevaral features(Bm and Pm)
        self.lr = 0.01 # learning rate
        self.gamma = 0.9 # reward_decay
        #self.epsilon_max =  0.9 # e_greedy, org 0.9
        self.epsilon = 0.8 # e_greedy,org 0.9
        self.replace_target_iter = 300 # target and evaluate net update interval
        self.memory_size = 500 # repay memory D size
        self.batch_size = 32 # minibatch size 
        #self.epsilon_increment = e_greedy_increment # greedy change
        #self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize repay memory [s, a, r, s_] as empty
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))# 2 state features(2*2) and action and reward

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # use sess to activate operation
        self.sess = tf.Session()

        # use global_variables_initializer() if there is tf.Variable、tf.get_Variable
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = [] #记录下来每一步的误差

        #if output_graph:
        #    # $ tensorboard --logdir=logs
        #    # tf.train.SummaryWriter soon be deprecated, use following
        #    tf.summary.FileWriter("logs/", self.sess.graph)

        # collection of actions
        actions_tmp = []
        actions_str_tmp = ''

        for i in range(self.M): #Create a string of actions
            actions_str_tmp = actions_str_tmp + str(i) #'012'
        
        # Create the colums(list of actions) of q_table by permutations
        ###for i in itertools.combinations(actions_str_tmp, 2):
        for i in itertools.permutations(actions_str_tmp, 2):
             # i = ('0', '1')('0', '2')('1', '0')('1', '2')('2', '0')('2', '1')
             # () is tuple, [] is list. Tupel's elements must be the same type.
            actions_tmp.append(''.join(i)) # Concatenate string array；'' is delimiter
            ###actions_tmp.append(''.join(i[::-1])) #Copy i but inverse 
        #self.actions_tmp = actions_tmp #action is a list of string actions = ['01','02','10','12','20','21']
        
        actions = [] # record all the 36 actions
        for i in range(2*self.M):
            for j in range(2*self.M):
                actions.append(actions_tmp[i] + actions_tmp[j])
        # actions = ['0101','0102','0110','0112','0120','0121',
        #            '0201','0202','0210','0212','0220','0221',
        #            '1001','1002','1010','1012','1020','1021',
        #            '1201','1202','1210','1212','1220','1221',
        #            '2001','2002','2010','2012','2020','2021',
        #            '2101','2102','2110','2112','2120','2121']
        # actions[0] and actions[1] are Bm change, actions[2] and actions[3] are Pm changes
        self.actions = actions

        # calculate umk
        self.umk = self.calculate_umk()


    """Functions to get reward"""
    
    # calculate the total fading
    def calculate_fading(self, dist):
        h_pl_db = 148.1 + 37.6 * math.log10(dist) # the path loss(dB)
        h_sh_db = np.random.randn() * 8 # the shadowing(log-normal distribution)(dB)
        # To get the normal distribution with mean u and variance x²: x*np.random.randn()+u

        # The envelope of the sum of two orthogonal Gaussian noise signals obeys the Rayleigh distribution.
        # Get two orthogonal Gaussian functions, one is the real part, another is the imaginary part.
        ray_array = np.random.randn(2) 
        h_small = math.pow(ray_array[0], 2)+ math.pow(ray_array[1], 2) # Pr=Pt*h²

        h_large = math.pow(10 , ( -( h_pl_db + h_sh_db) / 10.0)) # Convert minus log to regular form

        h_total = h_small * h_large # the total fading
        return h_total

    # channel capacity
    def calculate_rm(self, m):
        rm = self.Bm[m] * math.log2(1 + self.Pm[m] * self.Hm[m] / (self.N0 * self.Bm[m]))
        return rm

    # bm and rm
    def calculate_bm(self, m):
        bm = self.calculate_rm(m) / math.log2(self.M_quant)
        return bm

    # QP value
    def calculate_QP(self, m):
        QP = self.alpha * math.exp(self.beta * self.calculate_bm(m))
        return QP

    # SSIM value
    def calculate_SSIM(self, m):
        SSIM = self.a * math.pow(self.calculate_QP(m), self.b) + self.c
        return SSIM

    ### get qm
    # distortion rate
    def calculate_qm(self, m):
        qm = 1 - self.calculate_SSIM(m)
        return qm
    
    ### get pm
    # error rate
    def calculate_pm(self, m):
        Eb = self.Pm[m] * self.Hm[m] / self.calculate_rm(m) 
        Q = 1 - norm.cdf(math.pow(self.B * 2 * Eb / self.N0, 0.5)) # the inverse of the cumulative density function(Q function)
        pm = self.A * Q 
        return pm
    
    # reward fuction(total information with importance)
    def calculate_umk(self):
        umk = 0
        for m in range(self.M):
            for k in range(self.K):
                umk = umk + self.F[m][k] * (1 - self.calculate_pm(m)) * (1 - self.calculate_qm(m))
        return umk
    

    """ Use DQN to get the maximum of umk """

    #---------- Build evaluate net and target net----------- 
    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 当前状态为x行2列矩阵
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # 目标网络的Q值为x行36列矩阵 for calculating loss function
        
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            # first layer output is second layer input
            with tf.variable_scope('l1'):
                # get_variable('name',[shape],<initializer>)
                # w is n_features*n_l1, b is 1*n_l1
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)# relu is an activation fuction ReLU
            
            # second layer. collections is used later when assign to target net
            # second layer output is action
            with tf.variable_scope('l2'):
                # get_variable('name',[shape],<initializer>)
                # w is n_l1*n_actions, b is 1*n_actions
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
        
        # -----------define LOSS funtion to update the evaluate network--------- 
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        
        # ------optimize the loss function by using RMSProp (Like GD) learning rate = 0.01(eta)------
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        # placeholder 为占位符，只是定义该变量类型，并未给其赋值，需要用feed_dict给它进行字典赋值
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_') # 下一状态为x行2列矩阵
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    
    # ----------choose action by present state-----------
    def choose_action(self, observation): # 将当前状态输入神经网络，输出每一个action对应的Q值
        # to have batch dimension when feed into tf placeholder
        #observation_array = np.array(observation)
        observation = observation[np.newaxis, :] #将一维数据observation变成二维数据[[]]，从而能够被TensorFlow处理

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions in evaluate net
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            # np.argmax() get the max index 
        else:
            action = np.random.randint(0, self.n_actions) 
        # action似乎是指第几个动作，在这里我把action变成具体的字符串
        a = self.actions[action]
        return a, action
    
    # ---------evaluate net performs the action, and return the reward--------
    def step(self, action):
        if_restart = False
        #state_last = str(self.Bm) + str(self.Pm) #return the last state to the update_q_table function 
        ##try to not change the last state
        ##state_last_list = self.Bm #Save the last state to avoid the next state doesn't meet the condition.
        state_last = []
        for m in range(self.M):
            state_last.append(self.Bm[m])
        for m in range(self.M):
            state_last.append(self.Pm[m])
        state_last_array = np.array(state_last)

        reward = 0
        #Action is like '0101', action[0] is addtion; while action[1] is subtraction
        self.Bm[int(action[0])] = self.Bm[int(action[0])] + self.deltaB
        self.Bm[int(action[1])] = self.Bm[int(action[1])] - self.deltaB
        self.Pm[int(action[2])] = self.Pm[int(action[2])] + self.deltaP
        self.Pm[int(action[3])] = self.Pm[int(action[3])] - self.deltaP
        if self.Bm[int(action[1])] < self.B_min or self.Pm[int(action[3])] < self.P_min: 
            reward = -100
            #reward = -1
            if_restart = True
        else:
            umk_new = self.calculate_umk()
            #reward = (umk_new - self.umk) * 50
            reward = umk_new
            if reward < 0 :
                reward = reward * 4
                #reward = reward * 2
            else:
                reward = reward * 2
            #reward = umk_new
            self.umk = umk_new

        return state_last_array, reward, if_restart
    
    #--------- not satisfy the boundary condition, so back to the initialized state
    #def restart(self):
    #    self.Bm[0] = random.uniform(self.B_min, (self.B_total-2*self.B_min))
    #    self.Bm[1] = random.uniform(self.B_min, (self.B_total-self.Bm[0]-self.B_min))
    #    self.Bm[2] = self.B_total - self.Bm[0] - self.Bm[1]

    #    self.Pm[0] = random.uniform(self.P_min, (self.P_total-2*self.P_min))
    #    self.Pm[1] = random.uniform(self.P_min, (self.P_total-self.Pm[0]-self.P_min))
    #    self.Pm[2] = 10000 - self.Pm[0] - self.Pm[1]

    #    self.umk = self.calculate_umk()
    #    return
    
    def restart(self):
        for i in range(self.M):
            self.Bm[i] = self.B_total / self.M
            self.Pm[i] = self.P_total / self.M
        self.umk = self.calculate_umk()
        return
        

    # ----------define the repay memeory D(s, a, r, s_)----------
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'): # use if hasattr to check whether object has the attibute
            self.memory_counter = 0  #索引memory的行数

        transition = np.hstack((s, [a, r], s_)) #np.hstack 按列存储样本，每个样本为一个行向量

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size # 取余操作
        self.memory[index, :] = transition # 将这一行全部存下来

        self.memory_counter += 1 # 到下一行，从下一行开始存储
        return
    
    # ----------choose minibatch transitions from repay memory to calculate yj and LOSS-----------
    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)  # update the target net 
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32) # [0,1,2,3,4,5,......,31]
        eval_act_index = batch_memory[:, self.n_features].astype(int) #
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        # self.cost_his.append(self.cost)

        # increasing epsilon
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return
    
    #def _del_(self):
    #    print("*********release the object*************")

    #def plot_cost(self):
    #    import matplotlib.pyplot as plt
    #    plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    #    plt.ylabel('Cost')
    #    plt.xlabel('training steps')
    #    plt.show()










    





        




 

  

