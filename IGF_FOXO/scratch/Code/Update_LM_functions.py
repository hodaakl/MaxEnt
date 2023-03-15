## updating the lambda

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

def update_lambda(Error, old_lambda,alpha_cons = 0.05 ):#, alpha_power = 1): true_constraints , alpha_arr,
    # nCons = len(old_lambda)
    ## read lambda order of magnitude 
    # lambda_om = np.log10(old_lambda)
    # alpha_arr  = np.ones(len(old_lambda))*10**(lambda_om -1)

    # alpha_arr = np.ones(len(true_constraints))*10**(alphameans_p)
    # alpha_arr[int(len(true_constraints)/2):] = np.ones(int(len(true_constraints)/2))*10**(alphameans_p-2)

    # alpha_arr = alpha_arr[:nCons]

    Lambda = old_lambda + alpha_cons*(Error[:len(old_lambda)])

    return Lambda


def openfile(filename):
    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [col for col in reader]
    rows=np.array(rows)
    rows=rows.astype(np.float64)
    return rows
