# -*- coding: utf-8 -*-

import time

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt



def MLP(X, Y, inner_layers=[], iterations = 5000):
    t0 = time.time()
    
    def LM(MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac, Jhist, vfun, i, Mu):
           
        def xy_act(MLPw, MLPx, MLPy, vfun):
            for i in range(len(MLPw)): 
                tmp = np.matmul(MLPw[i],MLPx[i])
                # print(tmp)
                MLPy[i] = vfun[i].af(tmp)
                MLPdaf[i] = vfun[i].daf(tmp)
                try:
                    MLPx[i+1][1:,:] = MLPy[i]
                except:
                    pass
            return MLPx,MLPy,MLPdaf
        
        def gradient(MLPg, Jac, y, yhat, Mu):
            A = np.matmul(Jac,Jac.T)
            B = np.identity(A.shape[0])
            C = np.matmul(np.linalg.inv(A+B*Mu),Jac) # cambiar mu
            D = np.reshape((Y.T-MLPy[-1]).values,(Y.shape[0]*Y.shape[1],1))
            E = np.matmul(C,D) # actualización de pesos
            
            ini = 0 
            for i in np.arange(1,len(MLPw)+1):
                MLPg[-i] = E[ini:ini+MLPg[-i].size].reshape(MLPg[-i].shape)
                ini += MLPg[-i].size
            
            return MLPg 
        
        MLPx, MLPy, MLPdaf = xy_act(MLPw, MLPx, MLPy, vfun)
        
        J = 0 
        
        for i in range(MLPw[-1].shape[0]):   # iteration over 'y' outputs
            ini = 0 
            for l in range(len(MLPw)):     # iteration over layers
            
                if l == 0:
                    # primer parte
                    J += MSE_reg.J(Y[i], MLPy[-1][i])
                    MLPd[-1][i,:] = MSE_reg.dEdY(Y[i], MLPy[-1][i])*MLPdaf[-1][i] # delta 1 (cambiar a funcion de costo genérica) # Mod. funcion de costo 
                    Jac[MLPw[-1].shape[1]*i:MLPw[-1].shape[1]*(i+1), MLPy[-1].shape[1]*i:MLPy[-1].shape[1]*(i+1)] = MLPd[-1][i]*MLPx[-1] # jacobiano
                    ini += MLPw[-1].shape[0]*MLPw[-1].shape[1] 
                    # print(MLPw[-1].shape[1]*i,MLPw[-1].shape[1]*(i+1), MLPy[-1].shape[1]*i,MLPy[-1].shape[1]*(i+1))
                    
                elif l == 1:
                    # segunda parte
                    MLPd[-2] = np.matmul(np.reshape(MLPw[-1][i,1:],(-1,1)),np.reshape(MLPd[-1][i],(1,-1)))*MLPdaf[-2] # delta 2
                    for j in range(MLPd[-2].shape[0]):
                        Jac[ini + j*MLPx[-2].shape[0]:ini + (j+1)*MLPx[-2].shape[0], MLPy[-1].shape[1]*i:MLPy[-1].shape[1]*(i+1)] = MLPd[-2][j,:]*MLPx[-2]
                        # print(ini + j*MLPx[-2].shape[0],ini + (j+1)*MLPx[-2].shape[0], MLPy[-1].shape[1]*i,MLPy[-1].shape[1]*(i+1))
                    ini += MLPw[-2].shape[0]*MLPw[-2].shape[1]
                    
                else: 
                    # tercera parte
                    MLPd[-l-1] = np.matmul(MLPw[-l][:,1:].T,MLPd[-l])*MLPdaf[-l-1] # delta
                    for j in range(MLPd[-l-1].shape[0]):
                        # print(ini + j*MLPx[-l-1].shape[0],ini + (j+1)*MLPx[-l-1].shape[0] , MLPy[-1].shape[1]*i,MLPy[-1].shape[1]*(i+1))
                        Jac[ini + j*MLPx[-l-1].shape[0]:ini + (j+1)*MLPx[-l-1].shape[0], MLPy[-1].shape[1]*i:MLPy[-1].shape[1]*(i+1)] = MLPd[-l-1][j,:]*MLPx[-l-1]
                    ini += MLPw[-l-1].shape[0]*MLPw[-l-1].shape[1]
 
            
 
        ###### Actualización pesos por gradiente #####
        MLPg = gradient(MLPg, Jac, Y, MLPy[-1], Mu)   
        MLPw_copy = MLPw.copy()
        
        for i in range(len(MLPg)):
            MLPw_copy[i] = MLPw[i]-MLPg[i]  # Multiplicador nabla arbitrario
        
     
        ###### Estimar siguiente iteracion #######        
        _, yhat, _ = xy_act(MLPw_copy, MLPx, MLPy, vfun)
     
        ###### 
        Jhat = 0 
        for i in range(MLPw[-1].shape[0]):
            Jhat += MSE_reg.J(Y[i], MLPy[-1][i])
        if Jhat < J:
            return MLPw_copy, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac, J, Mu/10

        
        return MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac, J, Mu*10
    
    
    
    
    
    ##### Define layers and weights #####
    layers = [X.shape[1],Y.shape[1]] # X, inner layer1, il2, ..., iln, Y 
    if inner_layers:
        for i in range(len(inner_layers)):
            layers.insert(i+1,inner_layers[i])   

    MLPw = [np.random.rand(layers[i+1],layers[i]+1)*2-1 for i in range(len(layers)-1)] # initializing weights randomly
        
    MLPx = [np.ones((layers[i]+1,X.shape[0])) for i in range(len(layers)-1)]
    MLPy = [np.ones((layers[i+1],X.shape[0])) for i in range(len(layers)-1)]

    MLPdaf = [np.ones((layers[i+1],X.shape[0])) for i in range(len(layers)-1)]
    MLPd = [np.ones((layers[i+1],X.shape[0])) for i in range(len(layers)-1)]
    MLPg = [np.ones((layers[i+1],layers[i]+1)) for i in range(len(layers)-1)]
    Jac = np.zeros((np.sum([MLPw[i].shape[0]*MLPw[i].shape[1] for i in range(len(MLPw))]),Y.shape[0]*Y.shape[1])) 

    vfun = [Tanh]*(len(layers)-2)+[Linear]
    Jhist = np.zeros(iterations)
    
    MLPx[0][1:,:] = X.T 

    ##### xy actualization #####
    Mu = 0.5
    for i in range(iterations):
        MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac, Jhist[i], Mu = LM(MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac, Jhist, vfun, i, Mu)

        
            
    print('execution time: {} seconds'.format(time.time()-t0))

    return MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac, Jhist, vfun

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
        return np.ones(x.shape)  



################# Funciones de Costo ###################
class MSE_reg():
    def J(y, yhat):
        return np.sum((y.values.T-yhat)**2)
    def dJdY(y, yhat):
        pass
    def dEdY(y,yhat):
        return -np.ones(yhat.shape)
        
    
##### #####

X = pd.read_csv('X.csv', header = None)
Y = pd.read_csv('Y.csv', header = None)
# X = (X-X.mean(axis=0))/X.std(axis=0)
# Y = (Y-Y.mean(axis=0))/Y.std(axis=0)


# MLP(X,Y) # No inner layers
MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac, Jhist, vfun = MLP(X,Y, [7,5], iterations = 5000) # With inner layers



