# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


MLP = [2,2,2] # X, inner layer1, il2, ..., iln, Y 

MLPw = [np.ones((MLP[i],MLP[i+1]+1)) for i in range(len(MLP)-1)]
MLPy = [np.ones((MLP[i+1]+1,1)) for i in range(len(MLP)-1)]

def MLPy_act():
    np.dot(MLPw[0],MLPy[0])
print(MLPy)

for layer in MLP: 
    pass