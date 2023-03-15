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
path = spec_folder_onServer + '0907_output_MuS/'

# ------------------------------------------


# file_name_lambda = path + 'Lambdas.csv'

# if os.path.exists(file_name_lambda): 
#     print('Fetching lambda')
#     df_lambdas = pd.read_csv(file_name_lambda, sep = ',', header = None) 
#     data_lambdas = df_lambdas.to_numpy()
#     iteration, _ = data_lambdas.shape
#     iteration = iteration -1
#     Lambda = data_lambdas[-1,:]
# else:
#     print('lambda file doesnt exist')
# load the lagrange multipliers 
#
Lambda = np.array([ 4.67511516e-04,  9.01107541e-05,  4.87780459e-05, -2.65276820e-04,          1.60786772e-05, -4.21118772e-05,  2.89280452e-04, -2.09495802e-04,        2.70561424e-04, -6.61020443e-04, -3.36585376e-04, -1.62627438e-05,       -1.37113885e-04, -4.39157311e-04, -5.70741845e-04, -4.39041920e-04,  -2.68094808e-04, -3.59750403e-04, -3.92230231e-04, -3.51189750e-04, 1.29920650e-04, -5.13824756e-04, -3.65842055e-04, -7.43277348e-04,   1.75960687e-05, -1.88842512e-04, -3.34719577e-04,  1.46344056e-04,   1.47199393e-04, -5.24895254e-04,  1.99407375e-04,  2.90749773e-05,     3.58888379e-04,  2.68990432e-04,  1.76213706e-04, -7.65951598e-04,   -9.94143062e-04, -9.95779945e-04, -9.97800654e-04, -9.98416273e-04,     -9.99247607e-04, -9.99371129e-04, -9.94036537e-04, -9.95990852e-04,      -9.97708333e-04, -9.98332892e-04, -9.99140765e-04, -9.99325138e-04,     -9.93975447e-04, -9.95683462e-04, -9.97641940e-04, -9.98174171e-04,      -9.99009353e-04, -9.99212743e-04, -9.93377643e-04, -9.95591531e-04,      -9.97503662e-04, -9.98041893e-04, -9.98751482e-04, -9.98962834e-04,      -9.93761848e-04, -9.95628292e-04, -9.96804989e-04, -9.97174193e-04,       -9.98145876e-04, -9.98719985e-04, -9.93541083e-04, -9.95233382e-04,       -9.94361495e-04, -9.95182119e-04, -9.97698291e-04, -9.98230462e-04])
#
#
iteration = 2
print(f'iteration {iteration}')
print()
RunMCMC(outpath = path,Lagrange_multi = Lambda, N_chain=20*(10**3), ignore_steps=100, save_step = 50, iteration = iteration, param_range = 0.5, special_fn_label = 'convG_2')
#, params_filename = 'convG_params1.csv', moments_filename = 'convG_mom1.csv', acc_ratio_filename = 'convG_acc1.csv')



