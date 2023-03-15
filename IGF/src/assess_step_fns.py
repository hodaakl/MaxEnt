# from multiprocessing.sharedctypes import Value
import numpy as np 
import random 
from PredictionFunction_v2 import Get_Init_fn,FoxOn_preds_fn
import time 

def Cal_E(Abund_Vec, Lambda): 
    """Calculates the Energy of the state Lagrange_mutlipliers.Abundance vectors
    input:  Abund_Vec shape of (72)
            Lambda shape of (72)
    output: Energy scalar """
    Energy = np.dot(Abund_Vec,Lambda)
    return Energy 


def new_params(pars_old,  upperbound, lowerbound , beta =.05 , MaxNumChange = 14 ):
    """Returns new parameters """
    if MaxNumChange > len(pars_old):
        raise ValueError('MaxNumChange> len(vector) ; for this argument: minimum 1 and maximum len(vec)')

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


def params_cutoff(pars, upperbound , lowerbound  ):
    Flag_bounds = 1  #assume that it is within bounds and check if that is false then change to 0. 
    #Flag_range  = 0  # assume that it is out of range and check if in range change to 1. 
    
    idx1 = np.where(pars>upperbound)
    idx2 = np.where(pars<lowerbound)
    

    
    if len(idx1[0])>0 or len(idx2[0])>0:
        Flag_bounds = 0
        
    return Flag_bounds

def abund_cutoff(abd_vec, upperbound, lowerbound):
    Flag_bounds = 1  #assume that it is within bounds and check if that is false then change to 0. 
    #Flag_range  = 0  # assume that it is out of range and check if in range change to 1. 
    
    idx1 = np.where(abd_vec>upperbound)
    idx2 = np.where(abd_vec<lowerbound)
    

    
    if len(idx1[0])>0 or len(idx2[0])>0:
        Flag_bounds = 0
        
    return Flag_bounds

def Get_sensitivity_flag(p_last, p_first, thresh = .1, timethis = False):

    sens_flag = 1
    rat = 1- p_last/p_first
    if rat < thresh:
        sens_flag = 0
    return sens_flag



# def Get_sensitivity_flag(k,L=125*10**-3, t=90*60, thresh = .2, timethis = False, meth = 'BDF'):
#     """ gets the flag on the sensitivity imposed on the cell 
#     input:  k cellular parameters 
#             L ligand concentration pM 
#             thresh sensitivity threshold
#     Output: sens_flag = 0 fails sensitivity test 
#             sense_flag =1 passes the sensitivity test"""
#     sens_flag = 1 
#     if timethis == True: 
#         t0=time.time()
#     z0 = Get_Init_fn(k)
#     if timethis==True: 
#         print(f' getting initial condition {time.time()-t0} seconds' )
#     if timethis == True: 
#         t0=time.time()
#     mean, _ = FoxOn_preds_fn(k,L,t, z0, meth = meth)
#     if timethis==True: 
#         print(f' getting predictions for {t} seconds and {L} nM took {time.time()-t0} seconds' )
#     sens_rat = 1- (mean/z0[7])
#     if sens_rat < thresh:
#         sens_flag=0


#     return sens_flag

# def Get_Sense_Thresh(k,L=125*10**-3, t=90*60):
#     sense_rat 
