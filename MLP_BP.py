# -*- coding: utf-8 -*-

import time

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt



def MLP(X, Y, inner_layers=[], iterations = 5000):
    t0 = time.time()
    
    def GD(MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jhist, vfun):    #Gradient Descent
            
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
        
        MLPx,MLPy,MLPdaf = xy_act(MLPw,MLPx,MLPy,vfun)
        MLPd = deltas(MLPw, MLPdaf, MLPy, Y)
        MLPg = gradient(MLPd, MLPx)
        MLPw = w_act(MLPw, MLPg, 0.005)
        return(MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jhist)
 
    def BP(MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jhist, vfun):    #Back Propagation
        def xy_act(MLPw, MLPx, MLPy, vfun):
            for i in range(len(MLPw)): 
                MLPy[i] = vfun[i].af(np.matmul(MLPw[i],MLPx[i]))
                try:
                    MLPx[i+1][1:,:] = MLPy[i]
                except:
                    pass
            return MLPx,MLPy
        
        def inner_delta(w, d, daf): 
            d_ = np.matmul(w,d)*daf
            return d_
        
        
        def outer_delta(daf, y, Y):
            J = (y-Y)**2/2
            E = y-Y
            print(J.sum())
            d_ = -E*daf
            return d_

        
        def gradient(d, x):
            return np.matmul(d,x)
        
        def w_act(w, g, n):
            return w - n*g
        

        
        MLPx,MLPy = xy_act(MLPw,MLPx,MLPy,vfun)
        l = len(MLPdaf)
        for i in range(len(MLPw)): 
            if i == 0:
                MLPd[l-1] = outer_delta(MLPdaf[l-i-1],MLPy[-1],Y.T)
            else:
                MLPd[l-i-1] = inner_delta(MLPw[l-i][:,1:].T,MLPd[l-i],MLPdaf[l-i-1])
            MLPg[l-i-1] = gradient(MLPd[l-1-i], MLPx[l-1-i].T)
            MLPw[l-i-1] = w_act(MLPw[l-i-1], MLPg[l-i-1], 0.001)
     
        
        return(MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jhist)
    
       
    ##### Define layers and weights #####
    layers = [X.shape[1],Y.shape[1]] # X, inner layer1, il2, ..., iln, Y 
    if inner_layers:
        for i in range(len(inner_layers)):
            layers.insert(i+1,inner_layers[i])   

    # MLPw = [np.random.rand(layers[i+1],layers[i]+1)*2-1 for i in range(len(layers)-1)] # initializing weights randomly
    MLPw = [np.array([[ -0.082977995,  -0.154439273],
                    [0.220324493,  -0.103232526],
                    [-0.499885625,  0.038816734],
                    [ -0.197667427, -0.080805486],
                    [-0.353244109, 0.1852195],
                    [ -0.407661405,  -0.29554775],
                    [-0.313739789, 0.378117436]]),
            np.array([[-0.472612407,-0.359613061,	-0.186575822,	-0.414955789,	-0.401653166,	0.191877114,	-0.481711723,	-0.219556008],
                    [0.17046751,	-0.301898511,	0.192322616,	-0.460945217,	-0.078892375,	-0.184484369,	0.250144315,	0.289279328],
                    [-0.082695198,	0.300744569,	0.376389152,	-0.33016958,	0.45788953,	0.186500928,	0.488861089,	-0.396773993],
                    [0.058689828,	0.468261576,	0.394606664,	0.378142503,	0.033165285,	0.334625672,	0.248165654,	-0.052106474]]),
            np.array([[0.408595503,	-0.369971428,	-0.288371884,	-0.446637455,	0.089305537],
                    [-0.206385852,	-0.480633042,	-0.234453341,	0.074117605,	0.19975836],
                    [-0.212224661,	0.178835533,	-0.008426841,	-0.353271425,	-0.397665571]])]

        
    MLPx = [np.ones((layers[i]+1,X.shape[0])) for i in range(len(layers)-1)]
    MLPy = [np.ones((layers[i+1],X.shape[0])) for i in range(len(layers)-1)]

    MLPdaf = [np.ones((layers[i+1],X.shape[0])) for i in range(len(layers)-1)]
    MLPd = [np.ones((layers[i+1],X.shape[0])) for i in range(len(layers)-1)]
    MLPg = [np.ones((layers[i+1],layers[i]+1)) for i in range(len(layers)-1)]
    
    vfun = [Tanh]*(len(layers)-2)+[Linear]
    Jhist = np.zeros(iterations)
    
    MLPx[0][1:,:] = X.T 

    ##### xy actualization #####
    for i in range(iterations):
        MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jhist = GD(MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jhist, vfun)
        # MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jhist = BP(MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jhist, vfun)

        
            
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

X = pd.read_csv('X.csv', header = None)
Y = pd.read_csv('Y.csv', header = None)
# X = (X-X.mean(axis=0))/X.std(axis=0)
# Y = (Y-Y.mean(axis=0))/Y.std(axis=0)


# MLP(X,Y) # No inner layers
MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jhist, vfun = MLP(X,Y, [7,4], iterations = 5) # With inner layers


