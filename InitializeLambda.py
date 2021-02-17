import csv
import numpy as np 
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
outputfolder = '/blue/pdixit/hodaakl/output/MaxEnt_0217/Run1/'
Constraint_mu_only = True 


if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

    

file_name_lambda =outputfolder+ 'Lambdas.csv'

if Constraint_mu_only: 
    dim = 24
else: 
    dim = 48

Lambda = 0.001*np.random.rand(dim)


if os.path.exists(file_name_lambda): 
    print('file already exists')
    
else:
    
    with open(file_name_lambda, 'w') as new_file_lambda:
        csv_writer_lambda = csv.writer(new_file_lambda, delimiter = ',')
        csv_writer_lambda.writerow(Lambda)
        new_file_lambda.flush()