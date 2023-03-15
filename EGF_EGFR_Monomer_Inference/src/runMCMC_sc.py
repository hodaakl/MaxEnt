# Author : Hoda Akl 
# created : 2022.04.05
# last edited: 2022.10.13
import numpy as np 
import os 
import pandas as pd
from datetime import datetime
from markovchain import RunMCMC
from tools import write_output
noise_factor =2
path = f'/blue/pdixit/hodaakl/A1MAXENT_EGF/Percentile_Constraints_NonDimerized/output_noise/output_ada_egf_percentile_nf_{noise_factor}_20221013/'
ArraysPath = '/blue/pdixit/hodaakl/A1MAXENT_EGF/Percentile_Constraints_NonDimerized/ArraysForMaxEnt/'
file_name_lambda = path + '/Lambdas.csv'
file_name_log = path + 'log.csv' #Define log file , keeps parameters of run

if os.path.exists(file_name_lambda): 
    print('Fetching lambda')
    df_lambdas = pd.read_csv(file_name_lambda, sep = ',', header = None) 
    data_lambdas = df_lambdas.to_numpy()
    iteration, _ = data_lambdas.shape
    iteration = iteration -1
    Lambda = data_lambdas[-1,:]
else:
    raise ValueError('lambda file doesnt exist')
print(f'iteration {iteration}')
#### LOAD ARRAYS 
# the bin edges boundaries 
BinEdgesDict = np.load(f'{ArraysPath}BinEdges_9bins.npy', allow_pickle= True).item()
# abundance bounds 
AbundBounds = np.load(f'{ArraysPath}segfr_lims_10conds_0304.npy')
LowerLimAbund = AbundBounds[:,0]
UpperLimAbund = AbundBounds[:,1]
# parameter bounds 
LowerLimPars = np.load(f'{ArraysPath}Low_Pars_NoDimerazationModel_0301.npy')
UpperLimPars = np.load(f'{ArraysPath}high_Pars_NoDimerazationModel_0301.npy')
# Ligand concentration array 
Larr = np.load(f'{ArraysPath}EGFR_doses_10Conditions_20201116.npy')
nchain = 10**4
# record the parameters of this run in a log file
# if it is the first iteration, create the log file, if not edit the log file. 
string_tosave = f'{datetime.now()},iteration = iteration ,outpath = {path}, noise_factor = {noise_factor},\
        N_chain= {nchain}, save_step = 50, ignore_steps = 20,  param_range = 0.02, num_par_change = 5,\
            timethis=False, predfactor=1'
write_output(file_name_log, string_tosave ) # save the parameters 
# Run MCMC 
RunMCMC(outpath = path, Lagrange_multi = Lambda,  Cons_Bound_dict = BinEdgesDict, abund_low_lim= LowerLimAbund, abund_upp_lim =UpperLimAbund ,\
    params_upperbound = UpperLimPars , params_lowerbound = LowerLimPars , L = Larr, noise_factor = noise_factor,\
        N_chain= nchain, save_step = 50, ignore_steps = 20, iteration = iteration , param_range = 0.02, num_par_change = 5,\
            timethis=False, predfactor=1)