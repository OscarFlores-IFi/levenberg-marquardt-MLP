# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# OBL = lambda Sp,Inp,Pw,Pu,Pb,Sk,Ro : ((Sp - Inp) / ((1 - Pb) * Pw * (1 - 0.03 * Ro)) - 1) * 170 / (2 * (1 - Sk))
# P = lambda Bl,Sp,Inp,Pw,Pu,Pb,Sk,Ro : (Sp - Inp - Pw * (1 - 0.03 * Ro) * (1 + Bl * (1 - Sk) / 170) * (1 - Pb)) * Bl * Pu / (1 - Pb)*24

# Bl,Sp,Inp,Pw,Pu,Pb,Sk,Ro = 70,4900,3720,740,0.84,0.07,0.25,0

# OBL_s = OBL(Sp,Inp,Pw,Pu,Pb,Sk,Ro)
# P_s = P(OBL_s,Sp,Inp,Pw,Pu,Pb,Sk,Ro)

OBL= lambda x: ((x[1] - x[2]) / ((1 - x[5]) * x[3] * (1 - 0.03 * x[7])) - 1) * 170 / (2 * (1 - x[6]))
P = lambda x: (x[1] - x[2] - x[3] * (1 - 0.03 * x[7]) * (1 + x[0] * (1 - x[6]) / 170) * (1 - x[5])) * x[0] * x[4] / (1 - x[4])*24
# OBL = lambda x: x.sum(axis=1)
# P = lambda x: x.mean(axis=1)

rand = np.random.rand(1000,8)
rand[:,0]= rand[:,0]*30+80
rand[:,1]= rand[:,1]*100+4750
rand[:,2]= rand[:,2]*80+3720
rand[:,3]= rand[:,3]*40+720
rand[:,4]= rand[:,4]*0.02+0.84
rand[:,5]= rand[:,5]*0.02+.08
rand[:,6]= rand[:,6]*0.05+.25
rand[:,7][rand[:,7]>.5]=1
rand[:,7][rand[:,7]<=.5]=0


rdf = pd.DataFrame(rand)
OBL_s, P_s = OBL(rdf), P(rdf)

n_rdf = rdf
n_rdf[8] = OBL_s
n_rdf[9] = P_s

n_rdf.to_csv('deterministic.csv')
print(OBL_s, P_s)
print(rand)