#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np 
import random
import time
# sve_ivp
import os
import pandas as pd
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 12})
from Update_LM_functions import calculate_constraints, update_lambda, openfile
from PredictionFunctions import cell_pred_fn

on_mac = False
on_thinkpad = False 
on_hpg = True


# In[2]:


if on_mac == True: 
    data_path = "/Volumes/hodaakl/"
if on_thinkpad== True: 
    data_path = "//exasmb.rc.ufl.edu/blue/pdixit/hodaakl/"
if on_hpg == True: 
    data_path = "/blue/pdixit/hodaakl/"

# specify the project you are working on     
spec_folder = data_path + 'A5MCMC_IGF_FoxO/'


# In[6]:



def read_csv( path , index = 0, dat = 'cellpreds' ): 
    if dat == 'cellpreds':
        fn = path + f'cellpreds_{index}.csv'
    if dat == 'moments':
        fn = path + f'moments_{index}.csv'
    if dat == 'lambdas':
        fn = path + f'Lambdas.csv'
    if dat == 'params': 
        fn = path + f'params_{index}.csv'  
    if dat == 'variance': 
        
        fn = path + f'variance_{index}.csv'
    
    df = pd.read_csv(fn, sep = ',', header = None) 
    
    table = df.to_numpy()
    return table
    


# In[ ]:





# ### Comparing the means and the second moment of the simulation to the data 
# 1.  ✅Load the parameters of the iteration I want to examine 
# 2.  from the parameters get the mean and the variance of the cells 
# 3.  compare with the experimental data

# In[9]:


# 1. Loading the parameters 


folder_out ='0403_test4/' #    Lambda = old_lambda.copy() + alpha_arr*(Error)/true_constraints
path = spec_folder + folder_out

# get the best iteration 
err_fn = path +  'Errors.csv'
df = pd.read_csv(err_fn, sep = ',', header = None , ) 
err_np = df.to_numpy()
mean_err = np.mean(abs(err_np), axis = 1)
Best_iteration = np.argmin(mean_err)
print(Best_iteration)

# load the parameters 
param_np = read_csv(path = path, index = Best_iteration, dat = 'params')
print(param_np.shape)




# In[12]:


# 2. Get the means and the 2nd moment of the cells 

# determining the conditions 
times_arr = np.array([ 0,  6, 12, 24, 45, 60, 90])*60 #make it in seconds 
L  = np.array([10,15,20,50])*10**-3 #make it in nM
nConds = int(len(times_arr)*len(L))

# initializing the data matrix 
nC, nP = param_np.shape 
mu_mat = np.empty((nC, nConds))
s_mat  = np.empty((nC, nConds))
v_mat  = np.empty((nC, nConds))

# for each cellular parameters, get the mean and the variance , obtain the second moment from the variance 
for i in range(nC):
    k = param_np[i,:]
    mu, var = cell_pred_fn(k, times_arr, L)
    mu_mat[i,:] = mu
    v_mat[i,:]  = var
    s_mat[i,:]  = var + mu**2
    
    if i%1000==0:
        print(f'at cell {i}')
    


# In[15]:


np.save(f'mu_mat_iter{Best_iteration}.npy',mu_mat)
np.save(f'SecondMoment_mat_iter{Best_iteration}.npy',s_mat)
np.save(f'Var_mat_iter{Best_iteration}.npy',v_mat)



# In[ ]:




