# -*- coding: utf-8 -*-

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MLP(X, Y, inner_layers=[]):
    t0 = time.time()

    def af(x): # Activation Function 
        return 1/(1+np.exp(-x)) # Sigmoid
        # return np.tanh(x) # Tanh
    
    def daf(x): # Derivative of the Activation Function
        return af(x)*(1-af(x)) # Sigmoid
        # return 1-af(x)**2 # Tanh
        
    
    def xy_act(MLPw,MLPx,MLPy):
        for i in range(len(MLPw)-1): 
            tmp = np.matmul(MLPw[i],MLPx[i])
            MLPy[i] = af(tmp)
            MLPdaf[i] = daf(tmp)
            try:
                MLPx[i+1][1:,:] = MLPy[i]
            except:
                pass
        return MLPx,MLPy,MLPdaf
    
    def deltas(MLPw, MLPdaf, MLPy, Y): # Needed for backpropagation
        l = len(MLPdaf)
        J = (MLPy[-1]-Y.T)**2/2
        print(J.sum())
        E = MLPy[-1]-Y.T
        for i in range(l):
            if i==0:
                MLPd[l-i-1] = E*MLPdaf[l-i-1]
            else:
                MLPd[l-i-1] = np.matmul(MLPw[l-i][:,1:].T,MLPd[l-i])*MLPdaf[l-i-1]
        return MLPd 
    
    def gradient(MLPd, MLPx):
        for i in range(len(MLPg)):
            MLPg[i] = np.matmul(MLPd[i],MLPx[i].T)
        return MLPg
    
    def w_act(MLPw, MLPg, n):
        for i in range(len(MLPw)):
            MLPw[i] -= n*MLPg[i]
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

    MLPx[0][1:,:] = X.T 

    ##### xy actualization #####
    for i in range(50):
        MLPx,MLPy,MLPdaf = xy_act(MLPw,MLPx,MLPy)
        MLPd = deltas(MLPw, MLPdaf, MLPy, Y)
        MLPg = gradient(MLPd, MLPx)
        MLPw = w_act(MLPw, MLPg, 0.01)
        
            
    print(time.time()-t0)

    return MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg
 
        
    
##### #####

X = pd.read_csv('deterministic.csv',index_col=0).iloc[:,:-2].values
Y = pd.read_csv('deterministic.csv',index_col=0).iloc[:,-2:].values
X = (X-X.mean(axis=0))/X.std(axis=0)
Y = (Y-Y.mean(axis=0))/Y.std(axis=0)


# MLP(X,Y) # No inner layers
MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg = MLP(X,Y,[4,3]) # With inner layers



