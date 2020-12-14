"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.

The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.8.0
gym 0.10.5
"""
import multiprocessing
import threading
#import tensorflow as tf
import tensorflow.compat.v1 as tf # use tensorflow 1.0
tf.disable_v2_behavior() # forbidden tensorflow 2.0
import numpy as np
#import gym
import os
import shutil
import matplotlib.pyplot as plt
import random 
import math 
import itertools
from scipy.stats import norm

"""System parameters"""
# total power(mw) and bandwidth(Hz) must be changed in run_this file
P_total = 10000
B_total = 200
M = 3 # number of vehicles
K = 3 # number of fames of every vehicles

F = [[0.9, 0.9, 1], [0.5, 0.4, 0.5], [0.1, 0.2, 0]] # importance of each fames, range in[0,1] 
# F = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

Dm = [] # distance between each vehicles and edge server, range in [0,0.5km] (server cover area) 
for m in range(M):
    Dm.append(0.3)
        
N0 = math.pow(10, -174 / 10.0)  # noise power density(sigma_square) mW/Hz

B_min = B_total / 10.0 # lower boundary of Bm
P_min = P_total / 100.0 # lower boundary of Pm 

# model parameters
# use in the SSIM
# low
a = -4.247e-10
b = 5.1
c = 0.9521

# high
# a = -1.627e-8
# b = 4.243
# c = 0.9513

# use in the QP
alpha = 45.96
beta = -8.648e-5

# quantitative order
M_quant = 2

# modulation order e.g.:QPSK
M_modul = 4

# use in error rate
A = 2 * (1 - 1 / math.pow(M_modul, 0.5)) / math.log2(math.pow(M_modul, 0.5))
B = 3 * math.log2(math.pow(M_modul, 0.5)) / (M_modul - 1)

"""A3C hyperparameters"""
#GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count() # the number of thread equals to the number of cpu
MAX_EP_STEP = 200 # maximum step in one episode  org.200
MAX_GLOBAL_EP = 2000 # maximum of the total step  org.2000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10 # the interval of the change
GAMMA = 0.9
ENTROPY_BETA = 0.01 # try more actions(to expand our action range)
LR_A = 0.0001 # learning rate for actor(gradients) org.0.0001
LR_C = 0.001 # learning rate for critic(gradients) org.0.001
GLOBAL_RUNNING_R = [] 
GLOBAL_EP = 0 
#env = gym.make(GAME)
#N_S = env.observation_space.shape[0] # number of state in state space
#N_A = env.action_space.shape[0] # number of action in action space
#A_BOUND = [env.action_space.low, env.action_space.high] # bound of output action

N_S = 6 # number of state in state space(3Bm+3Pm)
N_A = 4 # number of action in action space(only change the B0,B1,P0,P1)

# bound of output action  # array([-2.], dtype=float32)
# the B0 and B1 is in one distribution
#def calculate_A_BOUND(Bm_0, Pm_0):
#    A_BOUND = [np.array([B_min, B_min, P_min, P_min]), np.array([(B_total-2*B_min), (B_total-Bm_0-B_min), (P_total-2*P_min), (P_total-Pm_0-P_min)])]
#    return A_BOUND

#A_BOUND = [np.array([B_min, B_min, P_min, P_min]), np.array([5*B_min, 3*B_min, 50*P_min, 30*P_min])]
A_BOUND = [np.array([B_min, B_min, P_min, P_min]), np.array([B_total-2*B_min, B_total-2*B_min, P_total-2*P_min, P_total-2*P_min])]

""" define the global and local actor-critics"""
class ACNet(object):
    ####------define the global and local network------####
    def __init__(self, scope, globalAC=None): 
        ###------define global network------###       
        if scope == GLOBAL_NET_SCOPE:   
            with tf.variable_scope(scope):
                # placeholder(dtype, shape=x*number_of_state, name)
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S') # global state
                self.a_params, self.c_params = self._build_net(scope)[-2:] # get global parameters
       
        ###------define local net, calculate loss------###
        else:   
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S') # local state
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A') # local action
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget') 

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                ##------calculate loss------##
                td = tf.subtract(self.v_target, self.v, name='TD_error') # critic net to minimum the error
                
                #------minimum critic net's loss function------#
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td)) # critic net's loss function

                #------get the action's distribution------#
                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4
                # normal distribution of parameters: mu is the mean, sigma is the std deviation.
                normal_dist = tf.distributions.Normal(mu, sigma)
                tf.print(mu)
                
                #------maximum actor net's loss function------#
                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his) # get logπ(a, s; theta_p_local)
                    exp_v = log_prob * tf.stop_gradient(td) # get logπ(a, s; theta_p_local)*(R-V(s; theta_p_v_local))
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v # use entropy as error to expand our exploratiom
                    self.a_loss = tf.reduce_mean(-self.exp_v) # maximize the expected value (minimize the minus)
                
                ##------choose action w.r.t local parameters------##
                with tf.name_scope('choose_a'):
                    #A_BOUND = calculate_A_BOUND(Bm[0],Pm[0]) 
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])
                    #self.A_B1 = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND_B1[0], A_BOUND_B1[1])
                    #self.A_P0 = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND_P0[0], A_BOUND_P0[1])
                    #self.A_P1 = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND_P1[0], A_BOUND_P1[1])

                ##------get gradients of a_loss and c_loss w.r.t local parameters------##
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
            
            ##------update global network's parameters by gradients and reset local network's parameters to global parameters------##
            with tf.name_scope('sync'):
                with tf.name_scope('pull'): # copy the global net's parameters to the local net
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'): # push the gradients to the global net
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
    
    ####------build the actor and critic network------#####
    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1) # get normal distribution with 0 mean and 0.1 std deviation
        # add a fully connected layer, tf.layers.dense(input, units, active function, initialize_convolution_kernel, name)
        
        ###------input state, get action's distribution------###
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la') # prepocess the state
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu') # get distribution's mean w.r.t state
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma') # get distribution's std deviation w.r.t state
        ###------input state, get state value------###
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc') # preprocess the state
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v') # get the state value
        
        ###------get the parameters------### 
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    ####------the actual update action (sess function)------#####
    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict) # push the gradients to the global net
    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op]) # copy the global net's parameters to the local net
    
    ####------the actual action (sess function)------####
    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})

""" define the threads' actual work"""
class Worker(object):
    ####------initialize environment and global network------####
    def __init__(self, name, globalAC):
        #self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC) # initialize the local ACNet w.r.t global parameters

        Bm_thread = [] # bandwidth of each vehicles, it's initialized to be eventually distributed
        for m in range(M):
            Bm_thread.append(B_total / M)
        #self.Bm_thread = Bm_thread        
        self.Bm_thread = [1/8 * B_total, 3/8 * B_total, 1/2 * B_total]
        
        Pm_thread = [] # power of each vehicles, it's initialized to be eventually distributed
        for m in range(M):
            Pm_thread.append(P_total / M)
        #self.Pm_thread = Pm_thread
        self.Pm_thread = [1/8 * P_total, 3/8 * P_total, 1/2 * P_total]

        Hm_thread = [] # total loss and fading of each vehicles
        for m in range(M):
            Hm_thread.append(self.calculate_fading(Dm[m]))
        self.Hm_thread = Hm_thread 

        self.best_umk = self.calculate_umk()
        self.best_Bm = self.Bm_thread
        self.best_Pm = self.Pm_thread

    """Global functions to get umk"""
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
        rm = self.Bm_thread[m] * math.log2(1 + self.Pm_thread[m] * self.Hm_thread[m] / (N0 * self.Bm_thread[m]))
        return rm

    # bm and rm
    def calculate_bm(self, m):
        bm = self.calculate_rm(m) / math.log2(M_quant)
        return bm

    # QP value
    def calculate_QP(self, m):
        QP = alpha * math.exp(beta * self.calculate_bm(m))
        return QP

    # SSIM value
    def calculate_SSIM(self, m):
        SSIM = a * math.pow(self.calculate_QP(m), b) + c
        return SSIM

    ### get qm
    # distortion rate
    def calculate_qm(self, m):
        qm = 1 - self.calculate_SSIM(m)
        return qm
    
    ### get pm
    # error rate
    def calculate_pm(self, m):
        Eb = self.Pm_thread[m] * self.Hm_thread[m] / self.calculate_rm(m) 
        Q = 1 - norm.cdf(math.pow(B * 2 * Eb / N0, 0.5)) # the inverse of the cumulative density function(Q function)
        pm = A * Q 
        return pm
    
    # reward fuction(total information with importance)
    def calculate_umk(self):
        umk = 0
        for m in range(M):
            for k in range(K):
                umk = umk + F[m][k] * (1 - self.calculate_pm(m)) * (1 - self.calculate_qm(m))
        return umk

    
    ####------input a, get s_, r------####
    def step(self, action):        
        #state_record = []
        #for m in range(M):
        #    state_record.append(self.Bm_thread[m])
        #for m in range(M):
        #    state_record.append(self.Pm_thread[m])
        
        #if_restart = False

        self.Bm_thread[0] = action[0]
        self.Bm_thread[1] = action[1]
        self.Bm_thread[2] = B_total - self.Bm_thread[0] - self.Bm_thread[1]
        self.Pm_thread[0] = action[2]
        self.Pm_thread[1] = action[3]
        self.Pm_thread[2] = P_total - self.Pm_thread[0] - self.Pm_thread[1] 
        
        if self.Bm_thread[2] < B_min or self.Pm_thread[2] < P_min: 
            reward = int(self.Bm_thread[2]) - int(B_min)+int(self.Pm_thread[2]) - int(P_min)# org.-10
            reward = reward/1000
            # reward = -int(math.pow(reward, 0.5))
            #if_restart = True
            #for i in range (M): # back to the last state
            #    self.Bm_thread[i] = state_record[i]
            #    self.Pm_thread[i] = state_record[i+3]
            self.Bm_thread = [1/8 * B_total, 3/8 * B_total, 1/2 * B_total]
            self.Pm_thread = [1/8 * P_total, 3/8 * P_total, 1/2 * P_total]

        else:
            for m in range(M):
                self.Hm_thread[m] = self.calculate_fading(Dm[m])
        
            reward = 0
            umk_new = self.calculate_umk()
            reward = umk_new * 100 # org. *100

            if umk_new >= self.best_umk:
                print("**************************arrangement found*******************************")
                print("       found new arrangemant 'Bm'       :  " + str(self.Bm_thread))
                print("       found new arrangemant 'Pm'       :  " + str(self.Pm_thread))
                print("   under this arrangement, the umk is   :  " + str(umk_new))

                # iteration of F & best_Bm
                self.best_Bm = self.Bm_thread
                self.best_Pm = self.Pm_thread
                self.best_umk = umk_new
        
        state_next = []
        for m in range(M):
            state_next.append(self.Bm_thread[m])
        for m in range(M):
            state_next.append(self.Pm_thread[m])
        state_next_array = np.array(state_next)

        return state_next_array, reward
    
    ####------define the thread's actual work------####
    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP # global variables: GLOBAL_RUNNING_R=[]  GLOBAL_EP=0
        total_step = 1 # count the total steps of action (global) 
        buffer_s, buffer_a, buffer_r = [], [], [] # record the s, a, r, similiar to the Repay Memory
        
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            # add system reset!!!!!,here
            self.Bm_thread = [1/8 * B_total, 3/8 * B_total, 1/2 * B_total]
            self.Pm_thread = [1/8 * P_total, 3/8 * P_total, 1/2 * P_total]

            state = []
            for m in range(M):
                state.append(self.Bm_thread[m])
            for m in range(M):
                state.append(self.Pm_thread[m])
            s = np.array(state) # record the Bm and Pm as an array

            #s = self.env.reset() # get the state 
            ep_r = 0 # sum up the total reward in one episode 
            
            ###------actor net: choose action, get reward and next state------###
            for ep_t in range(MAX_EP_STEP): # 200 steps in one episode
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)  # choose action
                #s_, r, done = self.step(a) # get next step and reward and if_it's_terminal
                s_, r = self.step(a) # get next step and reward

                done = True if ep_t == MAX_EP_STEP - 1 else False # until reach the maxmum of one episode's step
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8) # normalize
                #buffer_r.append(r)
                
                ###------critic net: get the state value------###
                if total_step % UPDATE_GLOBAL_ITER == 0 or done: # update global net and assign to local net
                    if done:
                        v_s_ = 0   # terminal：next reward is zero
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]  # get this state value 
                    buffer_v_target = []
                    for r in buffer_r[::-1]:   # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], [] 
                    self.AC.pull_global()

                s = s_
               
                total_step += 1
                # record running episode reward
                if done:  # 200 steps finished
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(ep_r) # ep_r is the sum of initial 200 steps' reward
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1], # print the last one of the GLOBAL_RUNNING_R[]
                          )
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA') # actor net's RMSProp optimizer 
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC') # critic net's RMSProp optimizer
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # define the global net
        
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))
    
    # multithreaded scheduler, call the multithreading
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads) # merge the workers, wait until all the threads(workers) finish their work, go next 

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

