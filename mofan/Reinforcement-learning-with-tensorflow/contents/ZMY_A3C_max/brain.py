"""
Initialize the system model and build the A3C model
"""
import numpy as np
import pandas as pd
import random 
import math 
from scipy.stats import norm

class System_A3C_Model:
     # total power(mw) and bandwidth(Hz) must be changed in run_this file
    def __init__(self, P_total = 10000, B_total = 200) :
        self.M = 3 # number of vehicles
        self.K = 3 # number of fames of every vehicles

        self.B_total = B_total
        self.P_total = P_total

        self.F = [[0.9, 0.9, 1], [0.5, 0.4, 0.5], [0.1, 0.2, 0]] # importance of each fames, range in[0,1] 

        Dm = [] # distance between each vehicles and edge server, range in [0,0.5km] (server cover area) 
        for m in range(self.M):
            Dm.append(0.3)
        self.Dm = Dm

        Pm = [] # power of each vehicles, it's initialized to be eventually distributed
        for m in range(self.M):
            Pm.append(self.P_total / self.M)
        self.Pm = Pm

        Bm = [] # bandwidth of each vehicles, it's initialized to be eventually distributed
        for m in range(self.M):
            Bm.append(self.B_total / self.M)
        self.Bm = Bm

        self.N0 = math.pow(10, -174 / 10.0)  # noise power density(sigma_square) mW

        self.B_min = self.B_total / 10.0 # lower boundary of Bm
        self.P_min = self.P_total / 1000.0 # lower boundary of Pm 
        
        Hm = [] # total loss and fading of each vehicles
        for m in range(self.M):
            Hm.append(self.calculate_fading(self.Dm))
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

        # modulation order
        self.M_modul = 4

        # use in error rate
        self.A = 2 * (1 - 1 / math.pow(self.M_modul, 0.5)) / math.log2(math.pow(self.M_modul, 0.5))
        self.B = 3 * math.log2(self.M_modul) / (self.M_modul - 1)


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

    # distortion rate
    def calculate_qm(self, m):
        qm = 1 - self.calculate_SSIM(m)
        return qm

    # error rate
    def calculate_pm(self, m):
        Eb = self.Pm[m] * self.Hm[m] / self.calculate_rm(m)        
        Q = 1 - norm.cdf(math.pow(self.B * 2 * Eb / self.N0, 0.5)) # the inverse of the cumulative density function(Q function)
        pm = self.A * Q 
        return pm
    
    # reward fuction(total information with importance)
    def calculate_ukm(self):
        umk = 0
        for m in range(self.M):
            for k in range(self.K):
                umk = umk + self.F[m][k] * (1 - self.calculate_pm(m)) * (1 - self.calculate_qm(m))
        return umk






    





        




 

  

