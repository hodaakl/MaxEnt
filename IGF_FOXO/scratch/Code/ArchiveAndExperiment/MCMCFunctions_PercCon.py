import numpy as np 
import random 
import time
import os
import csv
# from scipy.integrate import solve_ivp
from PredictionFunctions import  get_prec_preds
# from MomentEquations import MomentsDiff_Eq_fn
import pandas as pd
from scipy.special import psi
from scipy.stats import gamma 


times_arr = np.array([ 0,  6, 12, 24, 45, 60, 90])*60 #make it in seconds 
L  = np.array([10,15,20,50])*10**-3 #make it in nM


par_dict = {'par_name': ['k1', 'k2','k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k_tot_Akt', 'k_tot_foxo']
            , 'low_lim_log':np.array([2.75,-3.5,-2.25,-0.5,-0.75,-0.75,-3.5,-6.75,-6,-3.75,-2.25,-3,4.25,2])
            , 'high_lim_log': np.array([4.25,-2,-0.75,1,0.75,0.75,-2,-5.25,-4.5,-2.25,-0.75,-1.5,5.75,3.5])}

## 
highbound = par_dict['high_lim_log']
lowbound = par_dict['low_lim_log']

def RunMCMC(outpath, Lagrange_multi,Cons_Bound_dict, params_upperbound = highbound, params_lowerbound = lowbound,  N_chain= 10, save_step = 2, ignore_steps = 2, iteration = 1 , param_range = 0.1, num_par_change = 5, timethis=True, predfactor=1):#, nBins = 10):
    """Function that runs a Markov Chain Monte Carlo 
    Inputs: Lagrange_multi: np.array of len = len(constraints ) used in updatelambda file
            initial_k: initial set of parameters np.array of length 8 
            params_upperbound: np.array of upper bound of parameters
            params_lowerbound: np.array of lower bound of parameters
            N_chain: number of steps in the chain, integer 
            Iteration: which iteration of modifying the lagrange multiplier is the MCMC running on """
    
    # perc_arr = np.arange(1/nBins,1,1/nBins)
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
    # Lagrange_multi = Lagrange_multi # this was when we were scaling the lagrange multi
    # nCons = len(Lagrange_multi)
    k = initial_K
    # means_arr, var_arr = cell_pred_fn(k, times_arr, L ) 
    pred_curr , mu_curr, v_curr = get_prec_preds(Cons_Bound_dict, k, times_arr, L)*predfactor
    if timethis == True:
        t1 = time.time()
        print('Got Preds  in ', str(t1-t), 'seconds')
        t = time.time()
    
    E_curr = Cal_E(Abund_Vec=pred_curr, Lambda= Lagrange_multi)

    if timethis==True:
        t1 = time.time()
        print('got energy ', str(t1-t), 'seconds')
        t = time.time()

    a = 0 

    
    params_filename = outpath + f'/params_{iteration}.csv'
    CellPreds_filename =outpath + f'/cellpreds_{iteration}.csv'
    var_filename =outpath + f'/var_{iteration}.csv'
    mu_filename = outpath + f'/mu_{iteration}.csv'

    acc_ratio_filename =outpath + f'AccRatio.csv'

    rj_par = 0 # counter for rejection of parameters because cutoff     
    ss=0
    ts = time.time()
    for i in range(N_chain):
#         print(i)

        new_k = new_params(k, params_upperbound, params_lowerbound, beta = param_range, MaxNumChange = num_par_change )
        Flag_params = params_cutoff(new_k, params_upperbound, params_lowerbound)
        if Flag_params ==1 : 

            pred_new , mu_new, v_new = get_prec_preds(Cons_Bound_dict, new_k, times_arr, L)*predfactor

            E_new = Cal_E(Abund_Vec=pred_new, Lambda= Lagrange_multi)

            deltaE = E_new - E_curr 
            # print(deltaE)
            deltaE = np.array([deltaE], dtype = np.float128)[0] 
            
            # print(deltaE)
            prob = np.exp(-deltaE)
            A = min(1,prob)

            if random.random() < A : #With probability A do x[i+1] = x_proposed
                a = a+1 
                k = new_k.copy()
                # pred_curr = pred_new.copy()
                pred_curr = pred_new.copy()
                mu_curr = mu_new.copy()
                v_curr = v_new.copy()
                E_curr = E_new.copy()
        else: 
            rj_par+=1 
        
        
        
        
        si = i+1 
        if si%save_step ==0 :
            ss = ss+1 

            if ss > ignore_steps:
                # print('saving 1')
                if timethis==True:
                    t = time.time()
                
                if os.path.exists(CellPreds_filename): 
                    with open(CellPreds_filename, 'a') as add_file:

                        csv_adder = csv.writer(add_file, delimiter = ',')
                        csv_adder.writerow(pred_curr)
                        add_file.flush()
                else:
                    with open(CellPreds_filename, 'w') as new_file:

                        csv_writer = csv.writer(new_file, delimiter = ',')
                        csv_writer.writerow(pred_curr)
                        new_file.flush()


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

                if os.path.exists(var_filename): 
                    with open(var_filename, 'a') as add_file_var:
                        csv_adder_var = csv.writer(add_file_var, delimiter = ',')
                        csv_adder_var.writerow(v_curr)
                        add_file_var.flush()
                else:
                    with open(var_filename, 'w') as new_file_var:

                        csv_writer_var = csv.writer(new_file_var, delimiter = ',')
                        csv_writer_var.writerow(v_curr)
                        new_file_var.flush()



                if os.path.exists(mu_filename): 
                    with open(mu_filename, 'a') as add_file_mu:
                        csv_adder_mu = csv.writer(add_file_mu, delimiter = ',')
                        csv_adder_mu.writerow(mu_curr)
                        add_file_mu.flush()
                else:
                    with open(mu_filename, 'w') as new_file_mu:

                        csv_writer_mu = csv.writer(new_file_mu, delimiter = ',')
                        csv_writer_mu.writerow(mu_curr)
                        new_file_mu.flush()
                        
                if si%6000==0:
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
