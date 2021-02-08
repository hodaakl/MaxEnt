#!/usr/bin/env python
# coding: utf-8


import csv
import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
# from multiprocessing import Pool
# import multiprocessing
import time
# sve_ivp
from scipy.integrate import solve_ivp
from collections import defaultdict
# import concurrent.futures 
import os
# with open('names.csv')
import pandas as pd

outputfolder = '/blue/pdixit/hodaakl/output/MaxEnt_0204/Run1/'


    
def model(t, y, K ,L):
    """Model to be solved by the differential equation"""
    # y is of length 16 
    # K is a vector of length 20  - the first 17 are the parameters and the index 18 is protein abundance and the last two have to do with noise. 
#     t  = t*60 #converting time to seconds
    ns = 16
    K = np.asarray(K) #changing it to numpy array 
    K1 = K[:17] # rate constants 
    K2 = K[17:] #protein initial
    K1 = 10**K1
#     K[:17] = K1
    K  = np.concatenate((K1 ,K2),axis=0, out=None)
#         % define rates
#     % EGF binding to EGFR monomer
    k1      = K[0]
#     % EGF unbinding from EGFR
    kn1     = K[1]
#     % EGFR EGF-EGFR dimerization
    k2      = K[2]
#     % EGFR-EGF-EGFR undimerization
    kn2     = K[3]
#     % receptor phosphorylation
    kap     = K[4]
#     % receptor dephosphorylation
    kdp     = K[5]
#     % degradation of inactive
    kdeg    = K[6]
#     % degradation of active
    kdegs   = K[7]
#     % internalization of inactive
    ki      = K[8]
#     % internalization of active
    kis     = K[9]
#     % recycling of inactive
    krec    = K[10]
#     % recycling of active
    krecs   = K[11]
#     % rate of pEGFR binding to Akt
    kbind   = K[12]
#     % rate of pEGFR-Akt unbinding
    kunbind = K[13]
#     % Rate of Akt phosphorylation
    kpakt   = K[14]
#     % rate of pAkt dephosphorylation
    kdpakt  = K[15]
#     % EGFR delivery rate
    ksyn    = K[16]

#     % define concentrations

#     % ligand free receptors, plasma 
    r    = y[0]
#     % ligand free receptors, endosomes
    ri   = y[1]
#     % ligand bound receptors, plasma membrane
    b    = y[2]
#     % ligand bound receptors, endosomes
    bi   = y[3]
#     % 1 ligand bound dimers, plasma membrane
    d1   = y[4]
#     % 1 ligand bound dimers, endosomes
    d1i  = y[5]
#     % 2 ligand bound dimers, plasma membrane
    d2   = y[6]
#     % 2 ligand bound dimers, endosomes
    d2i  = y[7]
#     % 1 ligand bound phosphorylated dimers, plasma membrane
    p1   = y[8]
#     % 1 ligand bound phosphorylated dimers, endosomes
    p1i  = y[9]
#     % 2 ligand bound phosphorylated dimers, plasma membrane
    p2   = y[10]
#     % 2 ligand bound phosphorylated dimers, endosomes
    p2i  = y[11]
#     % 1L dimer bound to Akt
    p1a  = y[12]
#     % 2L dimer bound to Akt
    p2a  = y[13]
#     % pakt
    pakt = y[14]
#     % free akt
    akt  = y[15]
#     % Need to set one of the rate constants for thermodynamic consistency
    #initialize the differential equation variable
    dys = np.zeros(ns)
# %
#     % free receptors, plasma membrane
    dys[0]  = ksyn - k1*L*r + kn1*b - ki*r + krec*ri - k2*r*b + kn2*d1
#     % free receptors, endosomes
    dys[1]  = ki*r - krec*ri - kdeg*ri
#     % bound receptors, plasma membrane
    dys[2]  = k1*L*r - kn1*b - k2*r*b + kn2*d1 - 2*k2*b*b + 2*kn2*d2 - ki*b + krec*bi
#     % bound receptors, endosomes
    dys[3]  = ki*b - krec*bi - kdeg*bi
#     % 1 ligand bound dimer, plasma membrane
    dys[4]  = k2*r*b - kn2*d1 - kap*d1 + kdp*p1 - k1*L*d1 + kn1*d2 - ki*d1 + krec*d1i
#     % 1 ligand bound dimer, endosomes
    dys[5]  = ki*d1 - krec*d1i - kdeg*d1i + kdp*p1i - kap*d1i
#     % 2 ligand bound dimer, plasma membrane
    dys[6]  = k2*b*b - kn2*d2 - ki*d2 + krec*d2i - kap*d2 + kdp*p2 + k1*L*d1 - kn1*d2
#     % 2 ligand bound dimer, endosomes
    dys[7]  = ki*d2 - krec*d2i - kdeg*d2i + kdp*p2i - kap*d2i
#     % 1 ligand bound phosphorylated dimer, plasma membrane
    dys[8]  = kap*d1 - kdp*p1  - kis*p1 + krecs*p1i - k1*L*p1 + kn1*p2 - kbind*akt*p1 + kunbind*p1a + kpakt*p1a
#     % 1 ligand bound phosphorylated dimer, endosomes
    dys[9] = kis*p1 - krecs*p1i - kdegs*p1i + kap*d1i - kdp*p1i
#     % 2 ligand bound phosphorylated dimer, plasma membrane
    dys[10] = kap*d2 - kdp*p2 - kis*p2 + krecs*p2i + k1*L*p1 - kn1*p2 - kbind*akt*p2 + kunbind*p2a + kpakt*p2a
#     % 2 ligand bound phosphorylated dimer, endosomes
    dys[11]= kis*p2 - krecs*p2i - kdegs*p2i - kdp*p2i + kap*d2i
#     % p1 bound to Akt
    dys[12] = kbind*p1*akt - kpakt*p1a - kunbind*p1a
#     % p2 bound to Akt
    dys[13] = kbind*p2*akt - kpakt*p2a - kunbind*p2a
#     % pAkt
    dys[14] = kpakt*(p1a+p2a) - kdpakt*pakt
#     % free Akt
    dys[15] = -kbind*akt*(p1+p2) + kdpakt*pakt + kunbind*(p1a+p2a)
    return dys


def initial_conditions(K ):
# 
    """Initial conditions  - input (K) (parameters - len = 20) - Output y (16 species )"""
#     % 
#     % Initial conditions for individual variables
    K = np.asarray(K) #changing it to numpy array 
    K1 = K[:17] # rate constants 
    K2 = K[17:] #protein initial
#     L = K[0]
    K1 = 10**K1
#     K[:17] = K1
    K  = np.concatenate((K1 ,K2),axis=0, out=None)
#     % define rates
    #     % degradation of inactive
    kdeg    = K[6]
    #     % recycling of inactive
    krec    = K[10]
    #     % EGFR delivery rate
    ksyn    = K[16]
    #     % Total Akt abundance
    Akt0   = K[17] 
    #     % internalization of inactive
    ki      = K[8]
#     % surface receptors
    R0  = (kdeg + krec)*ksyn/(kdeg*ki)
#     % endosomal receptors
    R0i =  ksyn/kdeg

    y0s = np.zeros(16)
# 
#     % ligand free receptors, plasma membrane
    y0s[0]    = R0
#     % ligand free receptors, endosomes
    y0s[1]    = R0i
#     % free akt
    y0s[15]   = Akt0
    return y0s

def solve_model_atT_LSODA(K,L,tend):
    tspan = np.array([0,tend])
    y0 = initial_conditions(K)
    sol_dyn = solve_ivp(model, tspan, y0, method = 'LSODA', args=(K,L))
    sol = sol_dyn.y[:,-1]
    return sol

def calculate_constraints(data, ignore_steps = 0):
    """ inputs (data) with shape = (ncells, nConditions) : ncells would represent the number of MCMC samples taken"""
    mu = np.mean(data[ignore_steps:,:], axis = 0 ) # means along the column, to get the mean over all the cells
    s =  np.mean(data[ignore_steps:,:]**2, axis = 0 ) # means along the column, to get the mean over all the cells
    return [mu, s]
def calculate_energy(vec, Lambda):
    """returns energy. Input is an array of abundances of length 24"""
    energy = np.dot(vec, Lambda) #  + np.dot(vec**2, Lambda[n:])
    return energy 


def model_preds(K,L,t):
    """returns the aktp and segfr to be compared with experimental data"""
#     solving the model at t
    y = solve_model_atT_LSODA(K,L,t)
    aktp = y[14] + K[-2]
    segfr =  y[0] + y[2] + 2*(y[4] + y[6] + y[8]+ y[10] + y[12] + y[13] )
    segfr = segfr + K[-1]
    aktp = aktp/100
    segfr = segfr/100
    return [aktp, segfr]

def get_abund_vec(K): 
    nDis_AKT = 21
    nDis_EGFR = 3
    nDis = 24
    jj = 0
    
    akt_l = np.load('/blue/pdixit/hodaakl/Data/SingleCellData/AKT_L_21_training.npy')
    akt_t = np.load('/blue/pdixit/hodaakl/Data/SingleCellData/AKT_T_21_training.npy')
    akt_t = 60* akt_t

    segfr_l  = np.load('/blue/pdixit/hodaakl/Data/SingleCellData/SEGFR_L_3_training.npy')


    abund_curr = np.zeros(nDis) #holds total 
    abund_new = np.zeros(nDis)  # holds total

    for akt_dis_indx in range(nDis_AKT): 
        L = akt_l[akt_dis_indx]
        t = akt_t[akt_dis_indx]
        akt_model_curr, d =  model_preds(K= K, L = L, t = t)
        abund_curr[jj] = akt_model_curr
        jj = jj +1
        
    for egfr_dis_indx in range(nDis_EGFR):
        L = segfr_l[egfr_dis_indx]
        t = 180*60
        d, segfr_model_curr =  model_preds(K= K, L = L, t = t)
        abund_curr[jj] = segfr_model_curr
        jj = jj+1 
    
    return abund_curr


def new_pars_v2(pars_old,  upperbound , lowerbound, beta =.02 ):
    """Returns new parameters """
    # nchange is the number of parameters that will be updates (chosen at random)
    nchange = np.random.randint(1,11)
    
    delta = abs(upperbound - lowerbound )
    npoints = len(pars_old)
    
    idx = random.sample(range(0,npoints),nchange)
    
    newpars = pars_old.copy()
    
    newpars[idx] = newpars[idx] + beta*np.random.uniform(low = -delta[idx] , high =  delta[idx], size = (nchange,))
    idx1 = np.where(newpars>upperbound)
    idx2 = np.where(newpars<lowerbound)
    
    if len(idx1[0])>0 or len(idx2[0])>0:
        newpars = pars_old.copy()
    
    return newpars


# In[10]:


def RunSimulation_v3(Lambda,iteration, Nmc, ignore_steps =0, save_every_nsteps = 1):
    filename_abund = outputfolder + f'SS_data_{iteration}.csv'
    filename_pars  = outputfolder + f'Pars_{iteration}.csv'
#     Nmc = 4

    time0 = time.time()
    par_high = np.load('/blue/pdixit/hodaakl/Data/High_Pars_0130.npy')
    par_low = np.load('/blue/pdixit/hodaakl/Data/Lower_Pars_0130.npy')


      # number of mcmc steps to take to get a distribution 
    # Lambda = np.ones(nConstraints)
    SaveStep = save_every_nsteps
    
    ss = 0
    # Constraints = np.load('/blue/pdixit/hodaakl/Data/SingleCellData/Constraints_mu_s.npy')
    K_curr =  np.random.uniform(low = par_low , high = par_high) # current parameters 
    print('K_curr  = ' , K_curr)
    
    abund_curr = get_abund_vec(K_curr)
    
    E_curr = calculate_energy(vec =abund_curr ,Lambda = Lambda)
    a = 0 # for acceptance probility
    
    for i in range(Nmc): 
        print(f'step{i}')
#         print()
        K_new = new_pars_v2(pars_old = K_curr,  upperbound = par_high , lowerbound = par_low, beta = .2 )
        
        
        if len(np.where(K_new!=K_curr)[0])>0:    #K_new.any() != K_curr.any(): 
#             print('generated pars different from current pars')
#             print('k_new = ', K_new)
#             print('k_old = ', K_curr)
            abund_new = get_abund_vec(K_new)
            E_new  = calculate_energy(vec =abund_new ,Lambda = Lambda)
            A = min(1,np.exp(-E_new)/np.exp(-E_curr))
            if random.random() < A : #With probability A do x[i+1] = x_proposed
                a = a+1 
                K_curr = K_new.copy()
                abund_curr = abund_new.copy()
                E_curr = E_new.copy()

        if i%SaveStep ==0 :
            ss = ss+1 
            if ss > ignore_steps:
        # writing the single cell abundances 
                if os.path.exists(filename_abund): 
                    with open(filename_abund, 'a') as add_file:
                        csv_adder = csv.writer(add_file, delimiter = ',')
                        csv_adder.writerow(abund_curr)
                        add_file.flush()
                else:
                    with open(filename_abund, 'w') as new_file:

                        csv_writer = csv.writer(new_file, delimiter = ',')
                        csv_writer.writerow(fieldnames)
                        csv_writer.writerow(abund_curr)
                        new_file.flush()
#          writing the parameters
                if os.path.exists(filename_pars): 
                    with open(filename_pars, 'a') as add_file_pars:
                        csv_adder_pars = csv.writer(add_file_pars, delimiter = ',')
                        csv_adder_pars.writerow(K_curr)
                        add_file_pars.flush()
                else:
                    with open(filename_pars, 'w') as new_file_pars:

                        csv_writer_pars = csv.writer(new_file_pars, delimiter = ',')
                        csv_writer_pars.writerow(Par_fieldnames)
                        csv_writer_pars.writerow(K_curr)
                        new_file_pars.flush()

    A_Ratio = a/Nmc
    print(f'Acceptance ratio ={ A_Ratio}')
    time3 = time.time()
    print('time for one step = '+ str((time3 - time0)/Nmc))
    return print( f'one MCMC chain of iter {iteration} done')

Par_fieldnames = ['k1 ' ,'kn1  ','k2',' kn2 ','kap', 'kdp', 'kdeg', 'kdegs', 'ki', 'kis', 'krec', 'krecs', 'kbind', 'kunbind', 'kpakt', 'kdpakt', 'ksyn', 'totakt', 'aktbg', 'segfrbg']
fieldnames = ['pakt_L=0.003_t=0.0','pakt_L=0.1_t=5.0', 'pakt_L=0.1_t=15.0', 'pakt_L=0.1_t=30.0', 'pakt_L=0.1_t=45.0', 'pakt_L=0.32_t=5.0', 'pakt_L=0.32_t=15.0', 'pakt_L=0.32_t=30.0', 'pakt_L=0.32_t=45.0', 'pakt_L=3.16_t=5.0', 'pakt_L=3.16_t=15.0', 'pakt_L=3.16_t=30.0', 'pakt_L=3.16_t=45.0', 'pakt_L=10.0_t=5.0', 'pakt_L=10.0_t=15.0', 'pakt_L=10.0_t=30.0', 'pakt_L=10.0_t=45.0', 'pakt_L=100.0_t=5.0', 'pakt_L=100.0_t=15.0', 'pakt_L=100.0_t=30.0', 'pakt_L=100.0_t=45.0', 'segfr_L=0.0_t=180.0', 'segfr_L=1.0_t=180.0', 'segfr_L=100.0_t=180.0']

Nmc = 5000
ig_steps = 100
file_name_lambda = outputfolder + 'Lambdas.csv'
save_ev = 10

if os.path.exists(file_name_lambda): 
    df_lambdas = pd.read_csv(file_name_lambda, sep = ',', header = None) 
    data_lambdas = df_lambdas.to_numpy()
    iteration, _ = data_lambdas.shape
    iteration = iteration -1
    Lambda = data_lambdas[-1,:]
else:
    print('lambda file doesnt exist')
        
        
RunSimulation_v3(Lambda, iteration= iteration ,Nmc=Nmc,ignore_steps=ig_steps, save_every_nsteps= save_ev )