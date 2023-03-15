import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
import csv
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
path = spec_folder_onServer  + '0103_MuS_NewLM_newParRange/'

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
#
#
print(f'iteration {iteration}')
print()
RunMCMC(outpath = path,Lagrange_multi = Lambda,N_chain=2500, ignore_steps=20, save_step = 50, iteration = iteration, param_range = 0.2)
