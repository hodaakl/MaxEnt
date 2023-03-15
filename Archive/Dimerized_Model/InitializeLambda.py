import csv
import numpy as np 
import random
import time
from scipy.integrate import solve_ivp
from collections import defaultdict
import os
outputfolder = '/blue/pdixit/hodaakl/output/MaxEnt_0221/Run2/'
Constraint_mu_only = False 


if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)


    

file_name_lambda =outputfolder+ 'Lambdas.csv'

if Constraint_mu_only: 
    dim = 24
    Lambda_init = 0.001*np.random.rand(dim) 

else: 
    dim = 48
    Lambda_init = 0.001*np.random.rand(dim) 
    Lambda_init[24:] = 0.01*Lambda_init[24:] 



if os.path.exists(file_name_lambda): 
    print('file already exists')
    
else:
    
    with open(file_name_lambda, 'w') as new_file_lambda:
        csv_writer_lambda = csv.writer(new_file_lambda, delimiter = ',')
        csv_writer_lambda.writerow(Lambda_init)
        new_file_lambda.flush()