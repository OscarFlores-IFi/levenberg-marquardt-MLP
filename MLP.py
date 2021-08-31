# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def MLP(X, Y, inner_layers=[]):
    def activation_function(x):
        return 1/(1+np.exp(-x)) # Sigmoid
        # return np.tanh(x) # Tanh

    inner_layers = [2,3]
    
    ##### Define layers and weights #####
    layers = [X.shape[1],Y.shape[1]] # X, inner layer1, il2, ..., iln, Y 
    if inner_layers:
        for i in range(len(inner_layers)):
            layers.insert(i+1,inner_layers[i])   

    # MLPw = [np.ones((layers[i+1],layers[i]+1)) for i in range(len(layers)-1)]
    MLPw = [np.random.rand(layers[i+1],layers[i]+1) for i in range(len(layers)-1)]
    
    ##### Calculation of outputs #####
    nX = np.ones((X.shape[0],X.shape[1]+1))
    nX[:,1:] = X 
    
    y1 = np.matmul(MLPw[0],nX.T)
    
    # MLPx = [X.T
    # MLPy = [np.ones((layers[i+1],1)) for i in range(len(layers)-1)]
    
    # def MLPy_act():
    #     np.dot(MLPw[0],MLPy[0])
    return y1
    for layer in layers: 
        pass
        
    


X = pd.read_csv('deterministic.csv',index_col=0).iloc[:,:-2].values
Y = pd.read_csv('deterministic.csv',index_col=0).iloc[:,-2:].values




# MLP(X,Y) # No inner layers
y1 = MLP(X,Y,[4,3]) # With inner layers