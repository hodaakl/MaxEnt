# Author: Hoda Akl 
# Date:   05.04.2022 

import numpy as np 
import random 
import time
import pandas as pd
from PREDICTION_FUNCTIONS import  cell_pred_fn, get_prec_preds
from ASSESS_STEP import Cal_E, new_params, params_cutoff, abund_cutoff
from WRITE_OUTPUT import write_output


def RunMCMC(outpath, Lagrange_multi,  Cons_Bound_dict, abund_low_lim, abund_upp_lim,   params_upperbound , params_lowerbound , L,\
     noise_factor = 1, N_chain= 10, save_step = 2, ignore_steps = 2, iteration = 1 , param_range = 0.1, num_par_change = 5, \
         timethis=False, predfactor=1):#, integ_method = 'RK45'):#, nBins = 10):
    """Function that runs a Markov Chain Monte Carlo 
    Inputs: outpath: Path in which the files will be written
            Lagrange_multi: np.array of len = len(constraints ) used in updatelambda file
            Cons_Bound_dict: Bounds of the bins that are used as constraints 
            abund_low_lim: lower limit on accepted abundance 
            abund_upp_lim: Upper limit on accepted abundance
            params_upperbound: np.array of upper bound of parameters
            params_lowerbound: np.array of lower bound of parameters
            L: ligand cocentrations array that are used in the constraints # in nM 
            N_chain: number of steps in the chain, integer 
            save_step: save the MCMC step each save_steps 
            ignore_step: how many steps each save_step to ignore
            iteration: which iteration of modifying the lagrange multiplier is the MCMC running on
            param_range: parameter that defines how big the jump in parameters are in each step 
            num_par_change: maximum number of parameters allowed to change in each step 
            timethis: Timing different blocks of code to identify bottleneck, prints interval, default: False 
             """
    
    # perc_arr = np.arange(1/nBins,1,1/nBins)
    if timethis == True:
        t=time.time()	
    ## getting the initial_k 
    # print(f'order of mag for Lagrange_multi {np.log10(Lagrange_multi)}')
    # print(f'iteration {iteration}')
    if iteration == 0: 
        # print(' iteration index 0 , k_initial = avg(bounds)')
        print(f'iteration = 0, picking params as midpoint of bounds')
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
        print(f'Picking parameters from previous iteration - randomly chosen index {pickidx}')
        initial_K   = par_np[pickidx,:]    # 
    
    if timethis==True:
        t1 = time.time()
        print('picked params for first step in ', str(t1-t), 'seconds')
        t = time.time()
    # ------------------------ 
    # Lagrange_multi = Lagrange_multi # this was when we were scaling the lagrange multi
    # nCons = len(Lagrange_multi)
    k = initial_K  
    pred_curr , mu_curr, _ , v_curr = get_prec_preds(Cons_Bound_dict, k,  L, n= noise_factor)*predfactor
    Flag_abund, Flag_abund_ratio = abund_cutoff(mu_curr, abund_upp_lim, abund_low_lim)
    ### check if this passes the abundance flag 
    Flag_Cumu= Flag_abund_ratio*Flag_abund
    ApAb_count=0 #Appropriate Abundance counter 
    while Flag_Cumu==0:
        ApAb_count+=1  
        k = new_params(k, params_upperbound, params_lowerbound, beta = param_range, MaxNumChange = num_par_change )
        Flag_params, Flag_params_ratio = params_cutoff(k, params_upperbound, params_lowerbound)
        if Flag_params*Flag_params_ratio==1: 
            pred_curr , mu_curr, _, v_curr = get_prec_preds(Cons_Bound_dict, k, L)*predfactor
            
            Flag_abund, Flag_abund_ratio = abund_cutoff(mu_curr, abund_upp_lim, abund_low_lim)
    
        Flag_Cumu = Flag_abund* Flag_abund_ratio*Flag_params*Flag_params_ratio

    ### Exisiting this loop means we obtained out starting point and Flag_Cumu ==1 
    # print(ApAb_count)

    if timethis == True:
        t1 = time.time()
        print('Got Preds  in ', str(t1-t), 'seconds')
        t = time.time()
    
    E_curr = Cal_E(cell_preds= pred_curr, Lagrange_vec= Lagrange_multi)

    if timethis==True:
        t1 = time.time()
        print('got energy ', str(t1-t), 'seconds')
        t = time.time()

    ###### DEFINING COUNTER VARIABLES AND PATHS ####### 

    a = 0 # counter for accepted points
    rej_prob = 0 # counter for rejected points from probability
    rj_par = 0 # counter for rejection of parameters because cutoff     
    rj_abund=0 # counter for rejection of abundance because cutoff  
    ss=0 # counter for saving 
    # deltaE='notpresent' # initializing the var to avoid error in writing later

    # params_filename = outpath + 
    CellPreds_filename =outpath + f'/cellpreds_{iteration}.csv'
    # var_filename =outpath + f'/var_{iteration}.csv'
    # mu_filename = outpath + f'/mu_{iteration}.csv'
    info_filename =outpath + f'Info.csv'

    ###########################
    ts = time.time()
    for i in range(N_chain):
        # initialize the flags in case vars don't get defined in the if statement below
        Flag_abund = 0 
        Flag_abund_ratio = 0
        # Get new parameter vector 
        new_k = new_params(k, params_upperbound, params_lowerbound, beta = param_range, MaxNumChange = num_par_change )
        Flag_params, Flag_params_ratio = params_cutoff(new_k, params_upperbound, params_lowerbound)
        ### check all the constraints 
        if Flag_params*Flag_params_ratio==1: 
            pred_new , mu_new, _, v_new= get_prec_preds(Cons_Bound_dict, new_k,  L, n=noise_factor)*predfactor
            Flag_abund, Flag_abund_ratio  = abund_cutoff(mu_new, abund_upp_lim, abund_low_lim)

            if Flag_abund*Flag_abund_ratio==0:
                rj_abund+=1 

        else: 
            rj_par+=1 

        Flag_Cumu = Flag_abund*Flag_params*Flag_abund_ratio*Flag_params_ratio

        if Flag_Cumu==1: # it is within all bounds and constraints 

            E_new = Cal_E(cell_preds=pred_new, Lagrange_vec= Lagrange_multi)
            deltaE = E_new - E_curr 
            deltaE = np.array([deltaE], dtype = np.float128)[0] 
            # print(deltaE)
            prob = np.exp(-deltaE)
            A = min(1,prob)

            if random.random() < A : #With probability A do x[i+1] = x_proposed
                a = a+1 
                k = new_k.copy()
                pred_curr = pred_new.copy()
                mu_curr = mu_new.copy()
                v_curr = v_new.copy()
                E_curr = E_new.copy()
            else:
                rej_prob +=1 

        
        si = i+1 # counter for saving 
        if si%save_step ==0 :
            ss = ss+1 

            if ss > ignore_steps: 
                # print('saving 1')
                if timethis==True:
                    t = time.time()
                ### concatenate all calculated values for each cell to save in a single row
                outarr = np.concatenate((pred_curr, mu_curr, v_curr , k  ))
                write_output(CellPreds_filename, outarr )
                # commented out because this causes interference with other MCMC chains 
                # resulting in the rows being inconsistent between different chains . 
                # write_output(CellPreds_filename, pred_curr )
                # write_output(params_filename, k )
                # write_output(var_filename, v_curr )    
                # write_output(mu_filename, mu_curr )
                        
                if si%6000==0:
                    RJ_Ratio_Par = rj_par/si
                    RJ_Ratio_Ab = rj_abund/si
                    A_Ratio = a/si

                    # new line 
                    line = [iteration,'acc' , A_Ratio, 'rej_flag_par', RJ_Ratio_Par, 'rej_flag_abund', RJ_Ratio_Ab, 'deltaE', deltaE]
                    write_output(info_filename, line )         
        
                if timethis==True:
                    t1 = time.time()
                    print('saved to text files in  ', str(t1-t), 'seconds')
                    # t = time.time()    
        
    A_Ratio = a/N_chain
    RJ_Ratio_Par = rj_par/N_chain
    RJ_Ratio_Ab = rj_abund/N_chain
    # 
    line = [iteration,'acc' , A_Ratio, 'rej_flag_par', RJ_Ratio_Par, 'rej_flag_abund', RJ_Ratio_Ab ]
    # print(line)
    write_output(info_filename, line )
    tf = time.time()
    print(f'MCMC for each step in iteration {iteration} took {(tf-ts)/N_chain} seconds')
    print(f'Ratio of points rejected because they fall off abundance bounds = {RJ_Ratio_Ab}')
    print(f'Ratio of points rejected because they fall off parameter bounds = {RJ_Ratio_Par}')
    print(f'acceptance ratio = {A_Ratio}')
    print(f'Check: Sum up to 1? (sum ratios) = {A_Ratio+RJ_Ratio_Par+RJ_Ratio_Ab+rej_prob/N_chain  }')
    # print(f'Method of integration used for sensitivity constraint solver is {integ_method}')

    return(print(f'MCMC done'))