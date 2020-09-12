"""
build system model
"""
import numpy as np
import random
import numpy as np
import math
class System_model:
    def __init__(self, B_total = 2):
        self.M = 3 # numbers of vehicles
        self.N = 3 # numbers of categories
        self.stgShadow = 8 # std deviation; dB

        Delta = [] # vecters of \delta N dimentions all 1, weights of each categories
        for n in range(self.N):
            Delta.append(1)
        self.Delta = Delta
        V = [] # vecters of the speed of the N vehicles, slower than 60km/h
        for n in range(self.M):
            V.append(random.randint(0,60))
        self.V = V
        D = [] # vecters of the distance from each vehicles to the server; range from 0.1km to 1.1km
        for n in range(self.M):
            D.append(random.uniform(0.1,1.1))
        self.D = D
        S = [] # vecters of the transmit power for each vehicle; all 23dBm
        for n in range(self.M): 
            S.append(23)
        self.S = S
        Sigma_square = [] # vecters of the noise power for each vehicles; all -174 dbm/Hz
        for n in range(self.M):
            Sigma_square.append(-174)
        self.Sigma_square = Sigma_square
        self.fc = 2 # center frequency of the band; 2GHz

        self.delta_t = 200 # processing delay; 200ms

        self.te = 50 # small-scaling fading interval; 50ms

        self.B_total = B_total # total band resourse; MHz 2~20
        self.B_min = B_total/10.0
        self.P_min = 0.3 # minimum detection accuracy
        H = [] # the fading vecter  
        for i in range(self.M):
            H.append(self.calculate_fading(self.stgShadow, self.D[i]))
        self.H = H
        Bm = [] # band allocation
        for m in range(self.M):
            Bm.append(self.B_total / self.M)
        self.Bm = Bm
        

    def calculate_fading(self, stdShadow, dist):
        h_pl_db = 148.1 + 37.6 * math.log10(dist)
        h_sh_db = np.random.randn(1) * stdShadow

        ray_array = np.random.randn(2)
        h_small =math.pow((math.pow(ray_array[0],2)+ math.pow(ray_array[1],2)),0.5)

        h_large = math.pow(10 , ( ( - h_pl_db + h_sh_db) /10.0))

        h_combined = h_small * h_large
        return h_combined
    
    def calculate_R(self):
        
        return

