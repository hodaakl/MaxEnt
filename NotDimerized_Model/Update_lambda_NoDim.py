#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8


import csv
import numpy as np 
import random
import time
# sve_ivp
from scipy.integrate import solve_ivp
from collections import defaultdict
# import concurrent.futures 
import os
# with open('names.csv')
import pandas as pd



## Loads Lambda old. 
## Loads data 
## gets constraints data 
## gets new lambda 
## saves new lambda 



outputfolder = '/blue/pdixit/hodaakl/output/NoDim_Run1_0308/'
Constraint_mu_only = False 
SF = 0.00122 
ModelScaleFactor = SF  #this will be our scale factor
# 
mu_arr = np.load('/blue/pdixit/hodaakl/Data/SingleCellData/EGFR_mean_10Conditions_20201116.npy')
s_arr  = np.load('/blue/pdixit/hodaakl/Data/SingleCellData/EGFR_2ndMomentMean_10Conditions_20201116.npy' )
# L_arr  = np.load('/blue/pdixit/hodaakl/Data/SingleCellData/EGFR_doses_10Conditions_20201116.npy')

dim = 10 ## we got 10 dimensions 

Constraints = np.concatenate((mu_arr, s_arr), axis=0)

# to convert to a.u. to molecules
Constraints[:dim] = Constraints[:dim]/ModelScaleFactor
Constraints[dim:] = Constraints[dim:]/(ModelScaleFactor**2)



if Constraint_mu_only: 
    Constraints = Constraints[:dim]





def calculate_constraints(data, cons_mu_only = Constraint_mu_only ):
    """ inputs (data) with shape = (ncells, nConditions) : ncells would represent the number of MCMC samples taken"""
#     print('shape pf data ', data.shape)
    preds = np.mean(data, axis = 0 ) # means along the column, to get the mean over all the cells
    return preds 
# 
# 


def update_lambda(Error, old_lambda, alpha = 10**(-8), true_data = Constraints ):
    # alpha_array = alpha*np.ones(24)
    if len(Error) == dim: 
        alpha_arr = alpha*np.ones(dim)
    else: 
        alpha_arr = alpha*np.ones(2*dim)
        alpha_arr[dim:] = alpha_arr[dim:]*10**(-5)
    # Lambda = old_lambda.copy() + alpha*(Error)
    Lambda = old_lambda.copy() + alpha_arr*(Error)/true_data
    return Lambda


def openfile(filename):
    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [col for col in reader]
    rows=np.array(rows)
    rows=rows.astype(np.float128)
    return rows


file_name_lambda =outputfolder+ 'Lambdas.csv'
# lambdadf = pd.read_csv(lambda_path, sep = ',', header = None)
Lambda_np = openfile(file_name_lambda)


iterationp1, _ = Lambda_np.shape
iteration = iterationp1 -1



filename_abund = outputfolder + f'SS_data_{iteration}.csv'
df = pd.read_csv(filename_abund, sep = ',', header = None) 
Data_np = df.to_numpy()
### take away the nan values 
idxn = np.argwhere(np.isnan(Data_np))
idx_nan_rows = idxn[:,0]

data = np.delete(Data_np,idx_nan_rows, 0)



# data = openfile(filename_abund)

print(data.shape)
preds = calculate_constraints(data)
print(preds.shape)


# Constraints = Constraints[:24]
Preds = calculate_constraints(data)
# Preds = np.append(mu_sim, s_sim)
Error = Preds - Constraints 
print('avg abs error of iteration ' + str(iteration) + '=' + str(round(np.mean(abs(Error)),3)))
# rel_err = Error/Constraints
file_name_error = outputfolder+ 'Errors.csv'
# file_name_avg_abs_error =outputfolder+ 'Avg_abs_error.csv'

Old_Lambda = Lambda_np[-1,:]
# avgabserr = np.mean(abs(Error))
Lambda = update_lambda(Error = Error, old_lambda= Old_Lambda) 
Lambda= Lambda.tolist()
Error = Error.tolist()



#################################################### Storing the error 
if os.path.exists(file_name_error): 
    with open(file_name_error, 'a') as add_file_error:
        csv_adder_error = csv.writer(add_file_error, delimiter = ',')
        csv_adder_error.writerow(Error)
        add_file_error.flush()
else:
    with open(file_name_error, 'w') as new_file_error:

        csv_writer_error = csv.writer(new_file_error, delimiter = ',')
#         csv_writer_pars.writerow(Par_fieldnames)
        csv_writer_error.writerow(Error)
        new_file_error.flush()
    
#################################################### Storing the Lambda  

if os.path.exists(file_name_lambda): 
    with open(file_name_lambda, 'a') as add_file_lambda:
        csv_adder_lambda = csv.writer(add_file_lambda, delimiter = ',')
        csv_adder_lambda.writerow(Lambda)
        add_file_lambda.flush()
else:
    print('trouble loading file')


# In[ ]:




