# -*- coding: utf-8 -*-

import time

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt



def MLP(X, Y, inner_layers=[], iterations = 5000):
    t0 = time.time()
    
    def xy_act(MLPw, MLPx, MLPy, vfun):
        for i in range(len(MLPw)): 
            tmp = np.matmul(MLPw[i],MLPx[i])
            print(tmp)
            MLPy[i] = vfun[i].af(tmp)
            MLPdaf[i] = vfun[i].daf(tmp)
            try:
                MLPx[i+1][1:,:] = MLPy[i]
            except:
                pass
        return MLPx,MLPy,MLPdaf
    
    def deltas(MLPw, MLPdaf, MLPy, Y): 
        l = len(MLPdaf)
        J = (MLPy[-1]-Y.T)**2/2
        E = MLPy[-1]-Y.T
        # print(E)
        for i in range(l):
            if i==0:
                MLPd[l-i-1] = E*MLPdaf[l-i-1]
            else:
                MLPd[l-i-1] = np.matmul(MLPw[l-i][:,1:].T,MLPd[l-i])*MLPdaf[l-i-1]
        return MLPd, J.sum()
    
    def gradient(MLPd, MLPx):
        for i in range(len(MLPg)):
            MLPg[i] = np.matmul(MLPd[i],MLPx[i].T)
        return MLPg
    
    def w_act(MLPw, MLPg, n):
        # print(MLPw)
        for i in range(len(MLPw)):
            MLPw[i] = MLPw[i] - n*MLPg[i]
        return MLPw
    
    ##### Define layers and weights #####
    layers = [X.shape[1],Y.shape[1]] # X, inner layer1, il2, ..., iln, Y 
    if inner_layers:
        for i in range(len(inner_layers)):
            layers.insert(i+1,inner_layers[i])   

    # MLPw = [np.ones((layers[i+1],layers[i]+1)) for i in range(len(layers)-1)] # initializing weights at 1
    MLPw = [np.random.rand(layers[i+1],layers[i]+1)*2-1 for i in range(len(layers)-1)] # initializing weights randomly
    MLPx = [np.ones((layers[i]+1,X.shape[0])) for i in range(len(layers)-1)]
    MLPy = [np.ones((layers[i+1],X.shape[0])) for i in range(len(layers)-1)]

    MLPdaf = [np.ones((layers[i+1],X.shape[0])) for i in range(len(layers)-1)]
    MLPd = [np.ones((layers[i+1],X.shape[0])) for i in range(len(layers)-1)]
    MLPg = [np.ones((layers[i+1],layers[i]+1)) for i in range(len(layers)-1)]
    
    vfun = [Sigmoid]*(len(layers)-2)+[Linear]
    Jhist = np.zeros(iterations)
    
    MLPx[0][1:,:] = X.T 

    ##### xy actualization #####
    for i in range(iterations):
        print('iteration {}'.format(i))
        MLPx,MLPy,MLPdaf = xy_act(MLPw,MLPx,MLPy,vfun)
        MLPd, Jhist[i] = deltas(MLPw, MLPdaf, MLPy, Y)
        MLPg = gradient(MLPd, MLPx)
        MLPw = w_act(MLPw, MLPg, .001)
        
            
    print('execution time: {} seconds'.format(time.time()-t0))

    return MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jhist, vfun

############## Funciones de Activación ################
class Sigmoid():
    def af(x):
        return 1/(1+np.exp(-x)) # Sigmoid
    def daf(x):
        return Sigmoid.af(x)*(1-Sigmoid.af(x))
    
class Tanh():
    def af(x):
        return np.tanh(x)
    def daf(x):
        return 1-Tanh.af(x)**2

class Linear():
    def af(x):
        return x
    def daf(x):
        return 1       
    
##### #####

X = pd.read_csv('deterministic.csv',index_col=0).iloc[:,:-2].values
Y = pd.read_csv('deterministic.csv',index_col=0).iloc[:,-2:].values
X = (X-X.mean(axis=0))/X.std(axis=0)
Y = (Y-Y.mean(axis=0))/Y.std(axis=0)


# MLP(X,Y) # No inner layers
MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jhist, vfun = MLP(X,Y, [8,8,8], 5) # With inner layers



