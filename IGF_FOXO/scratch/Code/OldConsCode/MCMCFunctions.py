import numpy as np 
import random 
import time
import os
import csv
# from scipy.integrate import solve_ivp
from PredictionFunctions import  cell_pred_fn
# from MomentEquations import MomentsDiff_Eq_fn
import pandas as pd
from scipy.special import psi
from scipy.stats import gamma 


times_arr = np.array([ 0,  6, 12, 24, 45, 60, 90])*60 #make it in seconds 
L  = np.array([10,15,20,50])*10**-3 #make it in nM

par_dict = {'par_name': ['k1', 'k2','k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k_tot_Akt', 'k_tot_foxo']
            , 'low_lim_log':np.array([2.75,-3.5,-2.25,-0.5,-0.75,-0.75,-3.5,-6.75,-6,-3.75,-2.25,-3,4.25,2])
            , 'high_lim_log': np.array([4.25,-2,-0.75,1,0.75,0.75,-2,-5.25,-4.5,-2.25,-0.75,-1.5,5.75,3.5])}


def Cal_E(Abund_Vec, Lambda): 
    """Calculates the Energy of the state Lagrange_mutlipliers.Abundance vectors
    input:  Abund_Vec shape of (72)
            Lambda shape of (72)
    output: Energy scalar """
    Energy = np.dot(Abund_Vec,Lambda)
    return Energy 


def new_params(pars_old,  upperbound, lowerbound , beta =.05 , MaxNumChange = 14 ):
    """Returns new parameters """
    # pars_old is of length 8 
    npoints = len(pars_old)
    # nchange is the number of parameters that will be updates (chosen at random)
    nchange = np.random.randint(1,MaxNumChange)
#     print('changing nchange ', nchange, ' indices')
    delta = .5*abs(upperbound - lowerbound )
    
    
    idx = random.sample(range(0,npoints),nchange)
    
    newpars = pars_old.copy()

    newpars[idx] = newpars[idx] + beta*(2*np.random.rand(nchange)-1)*delta[idx]
    return newpars

def params_cutoff(pars, upperbound = par_dict['high_lim_log'], lowerbound = par_dict['low_lim_log'] ):
    Flag_bounds = 1  #assume that it is within bounds and check if that is false then change to 0. 
    #Flag_range  = 0  # assume that it is out of range and check if in range change to 1. 
    
    idx1 = np.where(pars>upperbound)
    idx2 = np.where(pars<lowerbound)
    

    
    if len(idx1[0])>0 or len(idx2[0])>0:
        Flag_bounds = 0
        
    return Flag_bounds

highbound = par_dict['high_lim_log']
lowbound = par_dict['low_lim_log']

## 
def RunMCMC(outpath, Lagrange_multi, norm_lagrange, percentile_constraint_left, percentile_constraint_right, params_upperbound = highbound, params_lowerbound = lowbound,  N_chain= 10,  save_step = 2, ignore_steps = 2, iteration = 1 , param_range = 0.1, num_par_change = 5, timethis=True):
    """Function that runs a Markov Chain Monte Carlo 
    Inputs: Lagrange_multi: np.array of len = len(constraints ) used in updatelambda file
            initial_k: initial set of parameters np.array of length 8 
            params_upperbound: np.array of upper bound of parameters
            params_lowerbound: np.array of lower bound of parameters
            N_chain: number of steps in the chain, integer 
            Iteration: which iteration of modifying the lagrange multiplier is the MCMC running on """
    if timethis == True:
        t=time.time()	
    ## getting the initial_k 
    # print(f'order of mag for Lagrange_multi {np.log10(Lagrange_multi)}')
    print(f'iteration {iteration}')
    if iteration == 0: 
        # print(' iteration index 0 , k_initial = avg(bounds)')
        
#         K_curr =  np.random.uniform(low = par_low/2 , high = par_high/2) # current parameters
        initial_K = .5*(params_upperbound + params_lowerbound ) 
        # E_curr = 10000

    else: 
        # print('picking the parameters from last iteration')
        # Load the parameters dataset from the previous iteration
        par_path =  outpath + f'/params_{iteration-1}.csv'
        par_df = pd.read_csv(par_path, header=None)
        par_np_all = par_df.to_numpy()
        ## Get the indices that are nan 
        idxn = np.argwhere(np.isnan(par_np_all))
        idx_nan_rows = idxn[:,0] # get the rows that have nan
        par_np = np.delete(par_np_all,idx_nan_rows, 0)
        maxidx   = par_np.shape[0]
        pickidx  = random.randrange(0,maxidx)
        initial_K   = par_np[pickidx,:]    # 
    
    if timethis==True:
        t1 = time.time()
        print('picked params for first step in ', str(t1-t), 'seconds')
        t = time.time()
    # ------------------------ 
    Lagrange_multi = Lagrange_multi/norm_lagrange
    # nCons = len(Lagrange_multi)
    k = initial_K
    means_arr, var_arr = cell_pred_fn(k, times_arr, L ) 
    if timethis == True:
        t1 = time.time()
        print('solved diff equations in ', str(t1-t), 'seconds')
        t = time.time()
    # means_arr = moments_pred_curr[:int(len(moments_pred_curr)/2)]
    # var_arr = moments_pred_curr[int(len(moments_pred_curr)/2):]
    # get the parameters of the gamma distribution 
    alpha_arr = (means_arr**2)/(var_arr)
    scale_arr = var_arr/means_arr
    ## now get the expectation of log(x) given beta and alpha 
    ## 
    lnx_exp_arr = np.log(scale_arr) + psi(alpha_arr)
    x25_arr = gamma.cdf(percentile_constraint_left, a =alpha_arr, scale = scale_arr)
    x75_arr = 1 - gamma.cdf(percentile_constraint_right, a =alpha_arr, scale = scale_arr)
    pred_curr  = np.concatenate((means_arr, lnx_exp_arr, x25_arr, x75_arr), axis = 0)

    if timethis==True:
        t1 = time.time()
        print('got preds ', str(t1-t), 'seconds')
        t = time.time()
    
    E_curr = Cal_E(Abund_Vec=pred_curr, Lambda= Lagrange_multi)

    if timethis==True:
        t1 = time.time()
        print('got energy ', str(t1-t), 'seconds')
        t = time.time()

    a = 0 

    
    params_filename = outpath + f'/params_{iteration}.csv'
    moments_filename =outpath + f'/cellpreds_{iteration}.csv'
    var_filename =outpath + f'/variance_{iteration}.csv'

    acc_ratio_filename =outpath + f'AccRatio.csv'

    rj_par = 0 # counter for rejection of parameters because cutoff     
    ss=0
    ts = time.time()
    for i in range(N_chain):
#         print(i)

        new_k = new_params(k, params_upperbound, params_lowerbound, beta = param_range, MaxNumChange = num_par_change )
        Flag_params = params_cutoff(new_k, params_upperbound, params_lowerbound)
        if Flag_params ==1 : 
            # print('new params are good ')

            means_arr,var_arr  = cell_pred_fn(new_k, times_arr, L ) 
            # if len(moments_pred_new) != nCons:
            #     moments_pred_new = moments_pred_new[:nCons]
            # print('moments*lambda')
            #print(moments_pred_new*Lagrange_multi)
            # print(f' order of mag for the moments pred {np.log10(moments_pred_new)}')
            
            # NOW WE HAVE MEANS AND SECOND MOMENT BUT WE NEED MEANS AND LN(X) 
            # COMMENT OUT IF ONLY MEANS AND SECOND MOMENTS ARE USED AS CONSTRAINTS 
            # means_arr = moments_pred_new[:int(len(moments_pred_new)/2)]
            # var_arr = moments_pred_new[int(len(moments_pred_new)/2):]
            # get the parameters of the gamma distribution 
            alpha_arr = (means_arr**2)/(var_arr)
            scale_arr = var_arr/means_arr
            ## now get the expectation of log(x) given beta and alpha 
            ## 
            lnx_exp_arr = np.log(scale_arr) + psi(alpha_arr)
            x25_arr = gamma.cdf(percentile_constraint_left, a =alpha_arr, scale = scale_arr)
            x75_arr = 1 - gamma.cdf(percentile_constraint_right, a =alpha_arr, scale = scale_arr)
            pred_new  = np.concatenate((means_arr, lnx_exp_arr, x25_arr, x75_arr), axis = 0)
            ## 
            # print('Predictions new means',means_arr)
            # print()
            # print('Predictions  new ln x',lnx_exp_arr)
            # print()
            # print('Predictions',pred_new)
            # print()
            # print()
            ###############

            # print('moments*lambda')
            # print(pred_new*Lagrange_multi)
            E_new = Cal_E(Abund_Vec=pred_new, Lambda= Lagrange_multi)

            deltaE = E_new - E_curr 
            # print(deltaE)
            deltaE = np.array([deltaE], dtype = np.float128)[0] 
            # print(f'deltaE = {deltaE}')
            # print(deltaE)
            prob = np.exp(-deltaE)
            #print(prob) 
            # if i%100==0:
            #     print(pred_new*Lagrange_multi)
            #     print(deltaE)
            #     print(prob)
            #     print()

            # print(prob)
            A = min(1,prob)

            if random.random() < A : #With probability A do x[i+1] = x_proposed
                a = a+1 
                k = new_k.copy()
                # pred_curr = pred_new.copy()
                pred_curr = pred_new.copy()
                E_curr = E_new.copy()
        else: 
            rj_par+=1 
        
        
        
        
        si = i+1 
        if si%save_step ==0 :
            ss = ss+1 
            if ss > ignore_steps:
                if timethis==True:
                    t = time.time()
                
                if os.path.exists(moments_filename): 
                    with open(moments_filename, 'a') as add_file:
                        csv_adder = csv.writer(add_file, delimiter = ',')
                        csv_adder.writerow(pred_curr)
                        add_file.flush()
                else:
                    with open(moments_filename, 'w') as new_file:

                        csv_writer = csv.writer(new_file, delimiter = ',')
                        csv_writer.writerow(pred_curr)
                        new_file.flush()

                if os.path.exists(var_filename): 
                    with open(var_filename, 'a') as add_file:
                        csv_adder = csv.writer(add_file, delimiter = ',')
                        csv_adder.writerow(var_arr)
                        add_file.flush()
                else:
                    with open(var_filename, 'w') as new_file:

                        csv_writer = csv.writer(new_file, delimiter = ',')
                        csv_writer.writerow(var_arr)
                        new_file.flush()
                        
                        
#          writing the parameters
                if os.path.exists(params_filename): 
                    with open(params_filename, 'a') as add_file_pars:
                        csv_adder_pars = csv.writer(add_file_pars, delimiter = ',')
                        csv_adder_pars.writerow(k)
                        add_file_pars.flush()
                else:
                    with open(params_filename, 'w') as new_file_pars:

                        csv_writer_pars = csv.writer(new_file_pars, delimiter = ',')
                        csv_writer_pars.writerow(k)
                        new_file_pars.flush()

                if si%1000==0:
                    RJ_Ratio = rj_par/si
                    A_Ratio = a/si
                    if os.path.exists(acc_ratio_filename): 
                        with open(acc_ratio_filename, 'a') as add_file_a:
                            csv_adder_a = csv.writer(add_file_a, delimiter = ',')
                            csv_adder_a.writerow([iteration,'acc' , A_Ratio, 'rej_flag_par', RJ_Ratio, 'deltaE', deltaE])
                       	    add_file_a.flush()
               	    else:
                        with open(acc_ratio_filename, 'w') as new_file_a:

                            csv_writer_a = csv.writer(new_file_a, delimiter = ',')
                            csv_writer_a.writerow([iteration, 'acc', A_Ratio, 'rej_flag_par', RJ_Ratio, 'deltaE', deltaE])
    #			    csv_writer_a.writerow([iteration, 'rej_par', RJ_Ratio])
                            new_file_a.flush()
        
                if timethis==True:
                    t1 = time.time()
                    print('saved to text files in  ', str(t1-t), 'seconds')
                    # t = time.time()    
        
    A_Ratio = a/N_chain
    RJ_Ratio = rj_par/N_chain

    # writing the acceptance ration 

    if os.path.exists(acc_ratio_filename): 
        with open(acc_ratio_filename, 'a') as add_file_a:
            csv_adder_a = csv.writer(add_file_a, delimiter = ',')
            csv_adder_a.writerow([iteration,'acc', A_Ratio, 'rej_flag_par', RJ_Ratio])
            add_file_a.flush()
    else:
        with open(acc_ratio_filename, 'w') as new_file_a:

            csv_writer_a = csv.writer(new_file_a, delimiter = ',')
            csv_writer_a.writerow([iteration, A_Ratio, 'rej_flag_par', RJ_Ratio])
            new_file_a.flush()
  
    tf = time.time()
    print(f'MCMC for each step in iteration {iteration} took {np.round(tf-ts)/N_chain}')
    print(f'acceptance ratio = {A_Ratio}')
    print(f'parameter rejected because of flag ratio = {RJ_Ratio}')
    return(print(f'MCMC done'))
