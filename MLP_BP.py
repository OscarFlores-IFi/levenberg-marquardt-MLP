# -*- coding: utf-8 -*-

import time

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt



def MLP(X, Y, inner_layers=[], iterations = 5000):
    t0 = time.time()
    
    def LM(MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac, Jhist, vfun, i):
           
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
        
        def gradient(MLPg, Jac, y, yhat):
            A = np.matmul(Jac,Jac.T)
            B = np.identity(A.shape[0])
            C = np.matmul(np.linalg.inv(A+B*0.5),Jac)
            D = np.reshape((Y.T-MLPy[-1]).values,(Y.shape[0]*Y.shape[1],1))
            E = np.matmul(C,D) # actualización de pesos
            
            ini = 0 
            for i in np.arange(1,len(MLPw)+1):
                MLPg[-i] = E[ini:ini+MLPg[-i].size].reshape(MLPg[-i].shape)
                ini += MLPg[-i].size
            
            return MLPg 
        
        MLPx, MLPy, MLPdaf = xy_act(MLPw, MLPx, MLPy, vfun)
        
        
        ######################################################################

        
        #        # reescribiendo jacobiano iterativo
        
        # for i in range(MLPw[-1].shape[0]):
            
        #     # primer parte
        #     print('primera parte')
        #     MLPd[-1][i,:] = MSE_reg.dEdY(Y[i], MLPy[-1][i])*MLPdaf[-1][i] # delta 1 (cambiar a funcion de costo genérica)
        #     Jac[MLPw[-1].shape[1]*i:MLPw[-1].shape[1]*(i+1), MLPy[-1].shape[1]*i:MLPy[-1].shape[1]*(i+1)] = MLPd[-1][i]*MLPx[-1] # jacobiano
        #     print(MLPw[-1].shape[1]*i,MLPw[-1].shape[1]*(i+1), MLPy[-1].shape[1]*i,MLPy[-1].shape[1]*(i+1))
            
        #     # segunda parte
        #     print('segunda parte')
        #     MLPd[-2] = np.matmul(np.reshape(MLPw[-1][i,1:],(-1,1)),np.reshape(MLPd[-1][i],(1,-1)))*MLPdaf[-2] # delta 2
        #     for j in range(MLPd[-2].shape[0]):
        #         ini = MLPw[-1].shape[0]*MLPw[-1].shape[1]
        #         Jac[ini + j*MLPx[-2].shape[0]:ini + (j+1)*MLPx[-2].shape[0], MLPy[-1].shape[1]*i:MLPy[-1].shape[1]*(i+1)] = MLPd[-2][j,:]*MLPx[-2]
        #         print(ini + j*MLPx[-2].shape[0],ini + (j+1)*MLPx[-2].shape[0], MLPy[-1].shape[1]*i,MLPy[-1].shape[1]*(i+1))
            
        #     # tercera parte
        #     print('tercera parte')
        #     MLPd[-3] = np.matmul(MLPw[-2][:,1:].T,MLPd[-2])*MLPdaf[-3] # delta 2
        #     for j in range(MLPd[-3].shape[0]):
        #         ini = MLPw[-1].shape[0]*MLPw[-1].shape[1] + MLPw[-2].shape[0]*MLPw[-2].shape[1]
        #         Jac[ini + j*MLPx[-3].shape[0]:ini + (j+1)*MLPx[-3].shape[0], MLPy[-1].shape[1]*i:MLPy[-1].shape[1]*(i+1)] = MLPd[-3][j,:]*MLPx[-3]
        #         print(ini + j*MLPx[-3].shape[0],ini + (j+1)*MLPx[-3].shape[0], MLPy[-1].shape[1]*i,MLPy[-1].shape[1]*(i+1))
            
            
        ######################################################################
        
        #        # reescribiendo jacobiano iterativo
        for i in range(MLPw[-1].shape[0]):   # iteration over 'y' outputs
            ini = 0 
            for l in range(len(MLPw)):     # iteration over layers
            
                if l == 0:
                    # primer parte
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
 
        
     


            
        MLPg = gradient(MLPg, Jac, Y, MLPy[-1])   
        
        for i in range(len(MLPg)):
            MLPw[i] = MLPw[i]-MLPg[i]*0.05
        
        
        
        
        
        
        
        
        
        
        
        return MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac
    
    
    
    
    
    ##### Define layers and weights #####
    layers = [X.shape[1],Y.shape[1]] # X, inner layer1, il2, ..., iln, Y 
    if inner_layers:
        for i in range(len(inner_layers)):
            layers.insert(i+1,inner_layers[i])   

    MLPw = [np.random.rand(layers[i+1],layers[i]+1)*2-1 for i in range(len(layers)-1)] # initializing weights randomly
    # MLPw = [np.array([[ -0.082977995,  -0.154439273],
    #                 [0.220324493,  -0.103232526],
    #                 [-0.499885625,  0.038816734],
    #                 [ -0.197667427, -0.080805486],
    #                 [-0.353244109, 0.1852195],
    #                 [ -0.407661405,  -0.29554775],
    #                 [-0.313739789, 0.378117436]]),
    #         np.array([[-0.472612407,-0.359613061,	-0.186575822,	-0.414955789,	-0.401653166,	0.191877114,	-0.481711723,	-0.219556008],
    #                 [0.17046751,	-0.301898511,	0.192322616,	-0.460945217,	-0.078892375,	-0.184484369,	0.250144315,	0.289279328],
    #                 [-0.082695198,	0.300744569,	0.376389152,	-0.33016958,	0.45788953,	0.186500928,	0.488861089,	-0.396773993],
    #                 [0.058689828,	0.468261576,	0.394606664,	0.378142503,	0.033165285,	0.334625672,	0.248165654,	-0.052106474]]),
    #         np.array([[0.408595503,	-0.369971428,	-0.288371884,	-0.446637455,	0.089305537],
    #                 [-0.206385852,	-0.480633042,	-0.234453341,	0.074117605,	0.19975836],
    #                 [-0.212224661,	0.178835533,	-0.008426841,	-0.353271425,	-0.397665571]])]

        
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
    for i in range(iterations):
        MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac = LM(MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac, Jhist, vfun, i)

        
            
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
MLPw, MLPx, MLPy, MLPd, MLPdaf, MLPg, Jac, Jhist, vfun = MLP(X,Y, [7,5,4], iterations = 5000) # With inner layers



