import csv
import numpy as np 
import random
import time
# sve_ivp
import os
import pandas as pd
import matplotlib.pyplot as plt
from os.path import dirname, realpath, sep, pardir
import sys
plt.rcParams.update({'font.size': 12})
sys.path.append('/blue/pdixit/hodaakl/A5MCMC_IGF_FoxO/Code/')
from UPDATE_LM_FNS import calculate_constraints, update_lambda, openfile
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
spec_folder = data_path + 'A5MCMC_IGF_FoxO/'
folder_out ='08302022_Ada_allconc_percentile/'

def Csv_to_Np(index = 0, dat = 'cellpreds' , path = spec_folder + folder_out ): 
    if dat == 'cellpreds':
        fn = path + f'cellpreds_{index}.csv'
    if dat == 'moments':
        fn = path + f'moments_{index}.csv'
    if dat == 'lambdas':
        fn = path + f'Lambdas.csv'
    if dat == 'params': 
        fn = path + f'params_{index}.csv'  
#     if dat == 'variance': 
        
        
#     fn = path + f'variance_{index}.csv'
    
    df = pd.read_csv(fn, sep = ',', header = None) 
    
    table = df.to_numpy()
    return table
path = spec_folder + folder_out
err_fn = path +  'Errors.csv'
df = pd.read_csv(err_fn, sep = ',', header = None , ) 
err_np = df.to_numpy()
print(err_np.shape)
nCons = err_np.shape[1]
real_cons = .1* np.ones(nCons)
rc_m= np.tile(real_cons[:len(err_np[0,:])] , [err_np.shape[0],1])
#print(rc_m.shape)
mean_err = np.mean(abs(err_np), axis = 1)
# mean_err.shape
real_abs = abs(err_np/rc_m)
mean_rel_abs = np.mean(real_abs, axis = 1)
# mean_err = np.mean(abs(err_np), axis = 1)
pathout = '/home/hodaakl/blue_pdixit/hodaakl/Figures/MaxEnt_0830'

plt.plot(range(len(mean_rel_abs) ), mean_rel_abs, label = 'mean')
median_rel_abs = np.median(real_abs, axis = 1)
# mean_err = np.mean(abs(err_np), axis = 1)
plt.plot(range(len(median_rel_abs) ), median_rel_abs, label = 'median')
plt.ylabel(' Error ')
plt.xlabel('Iteration')
plt.legend()
plt.tight_layout()

plt.savefig(pathout+'/Error.pdf')
print('saved figure')
plt.clf()
# save error array 
######################### MAKE MU and STD AND PARAMS matrices 
### LAST ITERATION 
last_iteration= len(mean_rel_abs)-1
print(last_iteration)
data = Csv_to_Np(index = last_iteration, path = path)
NumPars = 14 # number of parameters in this system
par_data = data[:,-NumPars:]

var_data = data[:,-NumPars-int(nCons/9):-NumPars]
mu_data = data[:,-NumPars-int(nCons/9)-int(nCons/9):-NumPars-int(nCons/9)]

# #### 
print(par_data.shape)
print(var_data.shape)
print(mu_data.shape)
### PLOTTING FOR THE PARAMETERS 
# mod = .25
par_dict = {'par_name': ['k1', 'k2','k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k_tot_Akt', 'k_tot_foxo']
            , 'low_lim_log':np.array([2.75,-3.5,-2.25,-0.5,-0.75,-0.75,-3.5,-6.75,-6,-3.75,-2.25,-3,4.25+.5,2])
            , 'high_lim_log': np.array([4.25,-2,-0.75,1,0.75,0.75,-2,-5.25,-4.5,-2.25,-0.75,-1.5,5.75+.5,3.5])
            , 'identifier': ['Receptor count ', 'Degredation of IGFR','Binding of IGFR to IGF','Unbinding IGFR to IGF ',
                      'Phosphorylation of bound receptor',  'Dephosphorylation of bound receptor ', 'Dephosphorylation of AKT',
                         'Phosphorylation of AKT', 'Phosphorylation of FoxO', 'Dephosphorylation of FoxO', 'Influx of FoxO to nucleus ',
                         'Efflux of FoxO from nucleus', 'k_tot_Akt', 'k_tot_foxo']}

# param_np = Csv_to_Np(index = 15, dat = 'params')
plt.clf()
plt.rcParams.update({'font.size': 12})
# fig, ax = plt.subplots()
pv = par_data[0,:]
ns, nk = par_data.shape
fig, axs = plt.subplots(2,7, figsize = (20,7))
pi = 0; pj = 0
for i in range(nk):
    axs[pi, pj].hist(par_data[:,i])
    axs[pi, pj].set_title(par_dict['identifier'][i])
    ll = par_dict['low_lim_log'][i]
    lh = par_dict['high_lim_log'][i]
    x_ll = np.ones(10)*ll
    x_lh = np.ones(10)*lh
    # p = np.ones(10)*pv[i]
    y = np.arange(0,200,200/10)
    axs[pi, pj].plot(x_ll, y, c = 'r')
    axs[pi, pj].plot(x_lh, y, c = 'r')
    # axs[pi, pj].plot(p, y, c = 'r')
    
    
    
#     axs[pi, pj].set_xlim([par_dict['low_lim_log'][i],  par_dict['high_lim_log'][i] ])
    pj+= 1
    if pj ==7:
        pj =0 
        pi +=1
    
plt.tight_layout()
pathout = '/home/hodaakl/blue_pdixit/hodaakl/Figures/MaxEnt_0830'

plt.savefig(pathout+'/parameters.pdf')
plt.show()
plt.clf()
print(f'saved parameter figure')
#### PLOTTING MU AGREEMENT 
MuVar_d = np.load(spec_folder + 'Arrays_for_max_ent/MuVar_dict_allconc.npy', allow_pickle=True).item()
print(MuVar_d)
Mu_Pop = np.mean( mu_data , axis = 0)
print(Mu_Pop.shape)
fig, axis = plt.subplots(figsize = (9,8))
times_arr = np.array([ 0,  6, 12, 24, 45, 60, 90])*60 #make it in seconds 
L  = np.array([10,15,20,25,50,250])*10**-3 #make it in nM
nT = len(times_arr)
nL = len(L)
nConds = int(nT*nL)
c = ['b','r','orange','g','k','purple', 'yellow', 'pink']
# means_sim_conc = np.array([])
i=0
for li in range(nL):
    plt.plot(times_arr/60, Mu_Pop[i:i+nT], label = f'Sim L = {L[li]} nM', c=c[li])
    plt.plot(times_arr/60, MuVar_d['mu'][i:i+nT], '--', label = f'Exp L = {L[li]} nM', c=c[li])
    
    
    i+=nT

plt.title('Population Means')
plt.xlabel('Min')
plt.ylabel('Nuclear FoxO')
plt.legend(bbox_to_anchor = [1,.5])
plt.tight_layout()

plt.savefig(pathout+'/means_agreement.pdf')
##############################################################
