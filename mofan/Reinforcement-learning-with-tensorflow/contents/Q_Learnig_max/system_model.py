"""
build system model
"""
import numpy as np
import pandas as pd
import random
import math
import itertools
class System_model:
    #initialize parameters
    def __init__(self, B_total=2e6): # total band resource is a variable parameter, range from 2 to 20 MHz
        self.M = 3 # numbers of vehicles
        self.N = 3 # numbers of categories

        Delta = [] # vectors of the weights of each categories, all 1
        for n in range(self.N):
            Delta.append(1)
        self.Delta = Delta
        
        ###Vm are used in calculating the small-scale fading by independent first-order autoregressive processes
        ###V = [] # vectors of the speed of the N vehicles, slower than 60km/h
        ###for n in range(self.M):
        ###    V.append(random.randint(0,60))
        ###self.V = V
        
        D = [] # vectors of the distance from each vehicles to the server; range from 0.1km to 1.1km
        for m in range(self.M):
            #D.append(random.uniform(0.1,1.1))
            D.append(3)
        self.D = D
        
        S = [] # vectors of the transmit power for each vehicle; all 23dBm
        for m in range(self.M): 
            S.append(math.pow(10 , 23/10.0))
        self.S = S
        
        Sigma_square = [] # vectors of the noise power for each vehicles; all -174 dBm/Hz
        for m in range(self.M):
            Sigma_square.append(math.pow(10 , -174/10.0))
        self.Sigma_square = Sigma_square
        
        ###self.fc = 2 # center frequency of the band; 2GHz

        ###self.delta_t = 200 # processing delay; 200ms

        ###self.te = 50 # small-scaling fading interval; 50ms

        self.B_total = B_total # total band resource; MHz 2~20; it's a variable parameter
        
        self.B_min = B_total/10.0 #minimum bandwidth limit
        
        ###self.P_min = 0.3 # minimum detection accuracy

        self.stdShadow = 8 # standard deviation; dB
        
        H = [] # the fading vector  
        for m in range(self.M):
            H.append(self.calculate_fading(self.stdShadow, self.D[m]))
        self.H = H
        
        Bm = [] # band allocation; it's initialized to be evenly distributed
        for m in range(self.M):
            Bm.append(self.B_total / self.M)
        self.Bm = Bm

        self.Rm_times = 300
        #model parameters; it's used in formula calculation
        self.am = [46.27, 45.96, 45.22]
        self.bm = [-7.086e-5, -8.648e-5, -1.052e-4]
        self.alpha_n = [-2.214e-12, -3.820e-13, -8.405e-8]
        self.beta_n = [6.741, 7.256, 4.158]
        self.gamma_n = [0.6940, 0.6958, 0.7250]

        #video clip information; it's used in formula calculation
        self.Imn = [[11.1, 1.6, 1.5], [1.0, 8.5, 0.3], [0.0, 1.9, 0.0]]

        #Q_learning parameters
        actions_tmp = []
        actions_str_tmp = ''

        for i in range(self.M): #Create a string of actions
            actions_str_tmp = actions_str_tmp + str(i) #'012'
        
        #Create the colums(list of actions) of q_table by permutations
        ###for i in itertools.combinations(actions_str_tmp, 2):
        for i in itertools.permutations(actions_str_tmp, 2):
             #i = ('0', '1')('0', '2')('1', '0')('1', '2')('2', '0')('2', '1')
             #() is tuple, [] is list. Tupel's elements must be the same type.
            actions_tmp.append(''.join(i)) #Concatenate string array；'' is delimiter
            ###actions_tmp.append(''.join(i[::-1])) #Copy i but inverse 
        self.actions = actions_tmp #action is a list of string actions = ['01','02','10','12','20','21']

        self.epsilon = 0.8 #greedy police
        self.q_alpha = 0.1 #learning rate
        self.q_gamma = 0.9 #discount factor
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) #build an empty q_table

        self.delta_B = B_total/100.0 #minimum change of the bandwidth (one step change)
       
        # F calculation  
        self.F = self.calculate_F()
    
    """Functions"""
    #Calculate the total fading
    def calculate_fading(self, stdShadow, dist):
        h_pl_db = 148.1 + 37.6 * math.log10(dist) #Get the path loss, which is usually minus(dB)
        h_sh_db = np.random.randn() * stdShadow #Get the shadowing(log-normal distribution)(dB)
        #TO get the normal distribution with mean u and variance x²: x*np.random.randn()+u

        #The envelope of the sum of two orthogonal Gaussian noise signals obeys the Rayleigh distribution.
        #Get two orthogonal Gaussian functions,one is the real part, and another is the imaginary part.
        ray_array = np.random.randn(2) 
        h_small =math.pow((math.pow(ray_array[0],2)+ math.pow(ray_array[1],2)),0.5)

        h_large = math.pow(10 , ( ( - h_pl_db + h_sh_db) /10.0)) #Convert log to regular form

        h_combined = h_small * h_large #the total fading
        return h_combined
    
    #Calculate the transfer rate: RTm=Bm*log2(1+Sm*hm/sigma²)
    def calculate_R(self, m):
        #transfer rate sets
        Rm_sets = []
        for cal_time in range(self.Rm_times):
            Rm_single = self.Bm[m]*math.log2(1+self.S[m]*self.calculate_fading(self.stdShadow,self.D[m])/(self.Sigma_square[m] *self.B_total/self.M ))
            Rm_sets.append(Rm_single)

        Rm = sum(Rm_sets)/self.Rm_times
        return Rm
    
    #Calculate the QP value of the video encoder: Qm=am*exp(bm*Rm) 
    def calculate_Q(self, m):
        #QP value
        Qm = self.am[m]*math.exp(self.bm[m]*self.calculate_R(m))
        return Qm 
    
    #Calculate the object detection accuracy: Pn=alpha_n*Q(beta_n)+gamma_n
    def calculate_P(self, m, n):
        #object detection accuracy
        Pn = self.alpha_n[n]*math.pow(self.calculate_Q(m),self.beta_n[n])+self.gamma_n[n]
        return Pn
    
    #Calculate the all quality assessment of content
    def calculate_F(self):
        #all quality assessment of content
        F = 0
        for m in range(self.M):
            for n in range(self.N):
                F = F + self.Delta[n]*self.Imn[m][n]*self.calculate_P(m,n)
        F = F/self.M
        return F
    

    """Use Q_learning to get the maximum of F"""
    
    #Enlarge the Q_Table's states.
    def check_state_exist(self, state):
        if state not in self.q_table.index: #Index is the states, while the column is the actions 
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state))

    #Choose actions (the best action(90%) or random(10%))
    def choose_action(self, observation): #observation is the present state
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            # choose the best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose one in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action  #'01' '10' '02' '20' '12' '21'

    # update state
    def step(self, action):
        if_restart = False
        state_last = str(self.Bm) #return the last state to the update_q_table function 
        ##try to not change the last state
        ##state_last_list = self.Bm #Save the last state to avoid the next state doesn't meet the condition.
        reward = 0
        #Action is like '01' '10' '02' '20' '12' '21', action[0] is addtion; while action[1] is subtraction
        self.Bm[int(action[0])] = self.Bm[int(action[0])] + self.delta_B
        self.Bm[int(action[1])] = self.Bm[int(action[1])] - self.delta_B
        if self.Bm[int(action[1])] < self.B_min:  #The restrictions include the minimum limit of Bm
            reward = -100
            if_restart = True

        else:
            F_new = self.calculate_F()
            reward = (F_new - self.F) * 50
            if reward < 0 :
                reward = reward * 4
            else:
                reward = reward * 2
            self.F = F_new

        return state_last, reward, if_restart
    
    #Back to the initialized state
    def restart(self):
        for i in range(self.M):
            self.Bm[i] = self.B_total / self.M
            self.F = self.calculate_F()
        return
        
    #update the q_table
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        #There is no terminal, since we hope the F could be higher.
        ###if s_ != 'terminal':
        q_target = r + self.q_gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        ###else:
        ###    q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.q_alpha * (q_target - q_predict)  # update
     
    



    

