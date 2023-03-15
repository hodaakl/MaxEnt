# Author: Hoda Akl 
# Date : 05.04.2022 

import csv
import numpy as np 

def calculate_constraints(data):
    """ Calculates the constrants: mean and 2nd moment of foxOnfor 36 conditions.
        Input:data : numpy matrix of shape "nCells X 72 
        output: pred_moments : predicted moments (mean and 2nd moments) from the data 
    """
    pred_moments = np.mean(data, axis = 0 ) # means along the column, to get the mean over all the cells
    
    return pred_moments 
# 

def update_lambda(Error, old_lambda,alpha_cons = 0.05):#, alpha_power = 1): true_constraints , alpha_arr,
    Lambda = old_lambda + alpha_cons*(Error[:len(old_lambda)])
    return Lambda

def update_lambda_adam(lg, m , v , der ,  beta1 =.8 , beta2  = .999, alpha = .1, i = 1, eps = 1e-8):
    """ This function uses ADAM to obtain new lagrange multipliers 
    lg: old lagrange multipliers 
    m , v, beta1, beta2, alpha, eps  : param used in adam algorithm
    der : derivative of the loss function  , in this case it is the error = preds - data 
    i : is the iteration -- used in adam algorithm
    """
    m = beta1*m + (1-beta1)*der
    v = beta2*v + (1-beta2)*der**2
    ##### unbias 
    mhat = m/(1-beta1**(i+1))
    vhat = v/(1-beta2**(i+1))
    lg = lg + alpha*mhat/(np.sqrt(vhat) + eps)
    return lg , m , v

def openfile(filename):
    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [col for col in reader]
    rows=np.array(rows)
    rows=rows.astype(np.float64)
    return rows
def adaptive_newparams(z,t,mz,mt, vz,vt, der_z, der_t, beta1 =.8 , beta2  = .999, alpha = .1, i = 1, eps = 1e-8):
    """Calculates the new parameters using the old parameters through ADAM algorithm"""
    mz = beta1*mz + (1-beta1)*der_z
    mt = beta1*mt + (1-beta1)*der_t
    vz = beta2*vz + (1-beta2)*der_z**2
    vt = beta2*vt + (1-beta2)*der_t**2
    ##### unbias 
    mzhat = mz/(1-beta1**(i+1))
    mthat  = mt/(1-beta1**(i+1))

    vzhat = vz/(1-beta2**(i+1))
    vthat = vt/(1-beta2**(i+1))
    
    # larr[i] = calc_loglikelihood(sigmas,g)
    # print(score)
    z = z + alpha*mzhat/(torch.sqrt(vzhat) + eps)
    t = t + alpha*mthat/(torch.sqrt(vthat) + eps)
    return z,t , mz, mt, vz, vt