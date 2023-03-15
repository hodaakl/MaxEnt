import numpy as np
# import random
import matplotlib.pyplot as plt
import time
import os
# import csv
import pandas as pd
from MARKOVCHAIN import RunMCMC
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
#
# specify the project you are working on     
spec_folder_onServer = data_path + 'A5MCMC_IGF_FoxO/'
foldername = '08302022_Ada_allconc_percentile/'
path = spec_folder_onServer + foldername 

# where are you saving 
# Arrays path 
arrays_path = spec_folder_onServer + 'Arrays_for_max_ent/'
# where are you saving 
save_locally=False
if save_locally == True: 
    path = foldername
    arrays_path = '/Users/hodaakl/Documents/github/MaxEnt_FoxO/Arrays_for_max_ent/'
# ------------------------------------------

# ------------------------------------------
# load the lagrange multipliers 
file_name_lambda = path + 'Lambdas.csv'

if os.path.exists(file_name_lambda): 
    print('Fetching lambda')
    df_lambdas = pd.read_csv(file_name_lambda, sep = ',', header = None) 
    data_lambdas = df_lambdas.to_numpy()
    iteration, _ = data_lambdas.shape
    iteration = iteration -1
    Lambda = data_lambdas[-1,:]
else:
    raise ValueError('lambda file doesnt exist')


# load the bounds of the bins
boundsdict = np.load(arrays_path + 'pcon_dict_allconc.npy', allow_pickle=True).item()

# load the max and min of the abundances MinFoxOBound.npy
abund_min_bound = np.load(arrays_path + 'MinFoxOBound_allconc.npy')
abund_max_bound = np.load(arrays_path + 'MaxFoxOBound_allconc.npy')

#define parameters bound
par_dict = {'par_name': ['k1', 'k2','k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k_tot_Akt', 'k_tot_foxo']
            , 'low_lim_log':np.array([2.75,-3.5,-2.25,-0.5,-0.75,-0.75,-3.5,-6.75,-6,-3.75,-2.25,-3,4.75,2])
            , 'high_lim_log': np.array([4.25,-2,-0.75,1,0.75,0.75,-2,-5.25,-4.5,-2.25,-0.75,-1.5,6.25,3.5])}
highbound = par_dict['high_lim_log']
lowbound = par_dict['low_lim_log']

## Define conditions of constraints
times_arr = np.array([ 0,  6, 12, 24, 45, 60, 90])*60 #make it in seconds 
L  = np.array([10,15,20, 25,50,250])*10**-3 #make it in nM

# define more vars
ncpc = len(boundsdict) # number constraints per condition
nc = int(len(times_arr)*len(L)) #number of conditions
nCons = int(nc*ncpc)
real_cons = .1*np.ones(nCons)
# ncons = len(real_cons)

print(f'MCMC.py reading iteration {iteration}')
print()
pf = 1
RunMCMC(path,Lambda,  boundsdict, abund_low_lim =  abund_min_bound,  abund_upp_lim= abund_max_bound ,\
    params_upperbound = highbound, params_lowerbound = lowbound,\
        L=L, times_arr=times_arr, iteration = iteration, \
            N_chain=7000 , save_step = 100, ignore_steps = 10, param_range = 0.08, \
            num_par_change = 5, timethis=False, predfactor = pf, cell_sense_thresh = .1)#, integ_method = 'BDF')#, nBins = 10)


# INTEGRATION METHODS 
# ‘RK45’ (default): Explicit Runge-Kutta method of order 5(4) [1]. The error is controlled assuming accuracy of the fourth-order method, but steps are taken using the fifth-order accurate formula (local extrapolation is done). A quartic interpolation polynomial is used for the dense output [2]. Can be applied in the complex domain.
# ‘RK23’: Explicit Runge-Kutta method of order 3(2) [3]. The error is controlled assuming accuracy of the second-order method, but steps are taken using the third-order accurate formula (local extrapolation is done). A cubic Hermite polynomial is used for the dense output. Can be applied in the complex domain.
# ‘DOP853’: Explicit Runge-Kutta method of order 8 [13]. Python implementation of the “DOP853” algorithm originally written in Fortran [14]. A 7-th order interpolation polynomial accurate to 7-th order is used for the dense output. Can be applied in the complex domain.
# ‘Radau’: Implicit Runge-Kutta method of the Radau IIA family of order 5 [4]. The error is controlled with a third-order accurate embedded formula. A cubic polynomial which satisfies the collocation conditions is used for the dense output.
# ‘BDF’: Implicit multi-step variable-order (1 to 5) method based on a backward differentiation formula for the derivative approximation [5]. The implementation follows the one described in [6]. A quasi-constant step scheme is used and accuracy is enhanced using the NDF modification. Can be applied in the complex domain.
# ‘LSODA’: Adams/BDF method with automatic stiffness detection and switching [7], [8]. This is a wrapper of the Fortran solver from ODEPACK.'
