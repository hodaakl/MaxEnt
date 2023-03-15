## Author: Hoda Akl 
## Date Created: 05/04/2022
#
# File contains functions that is needed in assessing one MCMC step. 
import numpy as np 
import random 

def Cal_E(cell_preds, Lagrange_vec): 
    """Calculates the Energy of the state Lagrange_mutlipliers.Cell_Preds
    input:  cell_preds : cellular predictions 
            Lagrange_vec : Lagrange multipliers
            Both vectors must match is size 
    output: Energy scalar """
    Energy = np.dot(cell_preds, Lagrange_vec)
    return Energy 


def new_params(pars_old,  upperbound  , lowerbound , beta =.02, MaxNumChange = 7 ):
    """Returns new parameters
    Input: pars_old: old parameter vector 
            upperbound: upper bound on parameter vector 
            lowerbound: lower bound on parameter vector 
            beta: parameter for how big the jump can be between parameters
            MaxNumChange: maximum number of parameters to change at once 
            
    Output: newpars: new parameter vector """
    if MaxNumChange > len(pars_old):
        raise ValueError('MaxNumChange> len(vector) ; for this argument: minimum 1 and maximum len(vec)')
    # pars_old is of length 7 
    # nchange is the number of parameters that will be updates (chosen at random)
    nchange = np.random.randint(1,MaxNumChange)
    # print('changing nchange ', nchange, ' indices')
    
    delta = .5*abs(upperbound - lowerbound )
    npoints = len(pars_old)
    
    idx = random.sample(range(0,npoints),nchange)
    # print('changing indexs', idx)
    
    newpars = pars_old.copy()
    
    newpars[idx] = newpars[idx] + beta*(2*np.random.rand(nchange)-1)*delta[idx]
    return newpars


def params_cutoff(pars, upperbound , lowerbound ):
    """Gives Flags to indicate the constraints on parameters 
    Input:  pars-> parameter vector 
            upperbound -> upper bound of parameters
            lowerbound-> lower bound of parameters
    Output: Flag_bounds -> 0 if out of the bounds, 1 if wintin it
            Flag_range  -> 0 is params fail ratio check, 1 if they pass """
    ## modified from before because here, we don't have k2 and kn2
#     pars = [ksyn, k1, kn1, kap, kdp, ki, kis]
    Flag_bounds = 1  #assume that it is within bounds and check if that is false then change to 0. 
    Flag_range  = 0  # assume that it is out of range and check if in range change to 1. 
    
    idx1 = np.where(pars>upperbound)
    idx2 = np.where(pars<lowerbound)
    
    Kd1 = 10**(pars[2]-pars[1]);  # kn1 - k1
#     Kd2 = 10**(pars[3]-pars[2]);  #kn2 - k2
    
    if len(idx1[0])>0 or len(idx2[0])>0:
        Flag_bounds = 0
    if Kd1>5 and Kd1<80:
        Flag_range = 1
        
    return Flag_bounds, Flag_range

def abund_cutoff(means, abunduppbound, abundlowerbound ,  SF = 0.00122 ):
    """ decides whether the solution is within the biological limits from the data 
    Input: means : solution of SEGFR for the 10 concentrations 
            abunduppbound: upper bound on abundances 
            abundlowerbound: lower bound on abundances 
    Output: abund_bound_flg : 0 if out of bounds, 1 if within bounds 
            fl_egfr : 0 if fails ratio check, 1 if passes it
    """
    abunduppbound = abunduppbound/SF
    abundlowerbound = abundlowerbound/SF
    ## Also scaled the bounds with SF
    
    if (len(np.where(means>abunduppbound)[0])>0)  or (len(np.where(means<abundlowerbound)[0])>0) :
        abund_bound_flg = 0
    else :
        abund_bound_flg = 1
    flEGFR_rat = ( means[-1])/( means[1] )
    
    if  flEGFR_rat < 1:
        fl_egfr = 1 
    else : 
        fl_egfr = 0
    
    
    return abund_bound_flg, fl_egfr
