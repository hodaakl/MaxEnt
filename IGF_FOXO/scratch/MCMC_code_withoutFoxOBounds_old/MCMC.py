import numpy as np
# import random
import matplotlib.pyplot as plt
import time
import os
# import csv
import pandas as pd
from MCMCFunctions import RunMCMC



# ------------------------------------------

## set path 
on_mac = False
on_thinkpad = False 
on_hpg = True

if on_mac == True: 
    data_path = "/Volumes/hodaakl/"
if on_thinkpad== True: 
    data_path = "//exasmb.rc.ufl.edu/blue/pdixit/hodaakl/"
if on_hpg == True: 
    data_path = "/blue/pdixit/hodaakl/"

# specify the project you are working on     
spec_folder_onServer = data_path + 'A5MCMC_IGF_FoxO/'
path = spec_folder_onServer + '0223_test/'

# ------------------------------------------

file_name_lambda = path + 'Lambdas.csv'

if os.path.exists(file_name_lambda): 
    print('Fetching lambda')
    df_lambdas = pd.read_csv(file_name_lambda, sep = ',', header = None) 
    data_lambdas = df_lambdas.to_numpy()
    iteration, _ = data_lambdas.shape
    iteration = iteration -1
    Lambda = data_lambdas[-1,:]
else:
    print('lambda file doesnt exist')
# load the lagrange multipliers 
#
# load the xlimits 
# 
xlimdict = np.load(spec_folder_onServer + 'Arrays_for_max_ent/XLimDict_25_75_020322.npy',allow_pickle='TRUE').item()
x25 = xlimdict['x25']
x75 = xlimdict['x75']

#
#
read_dictionary = np.load(spec_folder_onServer + 'Arrays_for_max_ent/cons_dict_mu_lnx_fraclr_020322.npy',allow_pickle='TRUE').item()
real_cons = read_dictionary['array']
ncons = len(real_cons)
mumu_data = np.mean(real_cons[:int(ncons/4)])
mulnx_data = np.mean(real_cons[int(ncons/4):int(ncons/2)])
mu_per = .25
# print(mumu_data)
# print(mus_data)
muc_len = 28 
lnx_len = 28 
pc_len = 28*2 
norm_vec = np.concatenate( (mumu_data*(np.ones(muc_len)), mulnx_data*(np.ones(lnx_len))  ,  mu_per*(np.ones(pc_len)) ), axis = 0)

print(f'iteration {iteration}')
print()
RunMCMC(outpath = path,Lagrange_multi = Lambda, norm_lagrange=norm_vec, N_chain=10000, ignore_steps=20, save_step = 50, iteration = iteration, param_range = 0.07, percentile_constraint_left=x25, percentile_constraint_right=x75, num_par_change = 6, timethis = False)
