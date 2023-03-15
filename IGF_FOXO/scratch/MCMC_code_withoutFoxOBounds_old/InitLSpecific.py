import os 
import pandas as pd 
import numpy as np 
import csv 
# ------------------------------------------
## set path 
on_mac = False
on_thinkpad = False 
on_hpg = True

if on_mac == True: 
    data_path = "/Volumes/hodaakl/"
if on_thinkpad== True: 
    data_path = "//exasmb.rc.ufl.edu/blue/pdixit/hodaakl/"
if on_hpg == True: 
    data_path = "/blue/pdixit/hodaakl/"

# specify the project you are working on     
spec_folder_onServer = data_path + 'A5MCMC_IGF_FoxO/'
path = spec_folder_onServer + '1029_output_MuS/'
# ------------------------------------------

# if not os.path.exists(path): 
#     os.mkdir(path)
# else: 
#     raise ValueError("folder already exists, can't initialize lambda")
# ------------------------------------------

constraint_mu_only = True 
lg_mu_OrderOfMag = -3 

# if constraint_mu_only == True: 
#     nCons = 36 
#     Lambda_init = np.random.rand(nCons)*10**(lg_mu_OrderOfMag+1)
# else: 
#     nCons = 72
#     Lambda_init = np.random.rand(nCons)*10**(lg_mu_OrderOfMag+1)
#     Lambda_init[int(nCons/2):] = Lambda_init[int(nCons/2):]*10**(-5)

# # save that lambda init 
# # ------------------------------------------


Lambda_init = np.array([8.05208763e-03, 7.06503270e-03, 8.85635186e-03, 7.58689013e-03, 1.76509982e-03, 3.31849690e-05, 2.96297390e-03, 5.26946999e-03,7.99519815e-03, 9.52357135e-03, 8.38811757e-03, 2.91999712e-03,  1.96247280e-03, 8.12100362e-03, 8.23929653e-03, 1.49104158e-03,    3.64511587e-03, 5.31026974e-03, 2.61773732e-03, 8.40026197e-04,   3.26366597e-03, 3.88145956e-03, 9.70191110e-03, 2.99158082e-03,    6.82023982e-03, 5.29864955e-03, 3.00984257e-04, 1.64929569e-03,     7.39713795e-03, 3.55933818e-03, 3.97665483e-03, 4.62336824e-03,   1.59344680e-03, 6.17054029e-04, 5.68251937e-03, 4.00367361e-03])

file_name_lambda =path+ 'Lambdas.csv'
with open(file_name_lambda, 'w') as new_file_lambda:
    csv_writer_lambda = csv.writer(new_file_lambda, delimiter = ',')
    csv_writer_lambda.writerow(Lambda_init)
    new_file_lambda.flush()