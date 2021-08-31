# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time

def MLP(X, Y, inner_layers=[]):
    t0 = time.time()
    
    def activation_function(x):
        return 1/(1+np.exp(-x)) # Sigmoid
        # return np.tanh(x) # Tanh
    
    def xy_act(MLPw,MLPx,MLPy):
        for i in range(len(MLPw)-1): 
            MLPy[i] = activation_function(np.matmul(MLPw[i],MLPx[i]))
        try:
            MLPx[i+1][1:,:] = MLPy[i]
        except:
            pass
        return MLPx,MLPy
    
    ##### Define layers and weights #####
    layers = [X.shape[1],Y.shape[1]] # X, inner layer1, il2, ..., iln, Y 
    if inner_layers:
        for i in range(len(inner_layers)):
            layers.insert(i+1,inner_layers[i])   

    # MLPw = [np.ones((layers[i+1],layers[i]+1)) for i in range(len(layers)-1)] # initializing weights at 0
    MLPw = [np.random.rand(layers[i+1],layers[i]+1)*2-1 for i in range(len(layers)-1)] # initializing weights randomly
    MLPx = [np.ones((layers[i]+1,X.shape[0])) for i in range(len(layers)-1)]
    MLPy = [np.ones((layers[i+1],X.shape[0])) for i in range(len(layers)-1)]

    MLPx[0][1:,:] = X.T 
    # for i in range(len(layers)-1): 
    #     MLPy[i] = activation_function(np.matmul(MLPw[i],MLPx[i]))
    #     try:
    #         MLPx[i+1][1:,:] = MLPy[i]
    #     except:
    #         pass
 
    ##### xy actualization #####
    MLPx,MLPy = xy_act(MLPw,MLPx,MLPy)
    
    
    print(time.time()-t0)
    return MLPw, MLPx, MLPy
 
        
    


X = pd.read_csv('deterministic.csv',index_col=0).iloc[:,:-2].values
Y = pd.read_csv('deterministic.csv',index_col=0).iloc[:,-2:].values




# MLP(X,Y) # No inner layers
MLPw, MLPx, MLPy = MLP(X,Y,[4,3]) # With inner layers