import os 
import pandas as pd 
import numpy as np 
import csv 
# ------------------------------------------
## set path 
###### IMPORTANT: ARE WE CONSTRAINING MEANS ONLY? FALSE IF NO, TRUE IF YES! 
constraint_mu_only = False 


on_mac = False
on_thinkpad = False 
on_hpg = True

if on_mac == True: 
    data_path = "/Volumes/hodaakl/"
#     spec_folder_onServer = data_path + 'Max/'
if on_thinkpad== True: 
    data_path = "//exasmb.rc.ufl.edu/blue/pdixit/hodaakl/"
if on_hpg == True: 
    data_path = "/blue/pdixit/hodaakl/"

# specify the project you are working on   
spec_folder_onServer = data_path + 'A5MCMC_IGF_FoxO/'
path = spec_folder_onServer + '0211_test1/'
# ------------------------------------------

if not os.path.exists(path): 
    os.mkdir(path)
else: 
    raise ValueError("folder already exists, can't initialize lambda")
# ------------------------------------------

read_dictionary = np.load(spec_folder_onServer + 'Arrays_for_max_ent/cons_dict_mu_lnx_fraclr_020322.npy',allow_pickle='TRUE').item()
real_cons = read_dictionary['array']
nCons = len(real_cons)
print('length of constraints array = ', nCons)
# real_cons = np.load(spec_folder_onServer + 'Arrays_for_max_ent/Cons_Arr_Means_SecMoment_72Scaled.npy')

Full_lambda_init = np.zeros(len(real_cons))
# Full_lambda_init[int(len(real_cons)/2):] = np.ones(int(len(real_cons)/2))*10**(-6)
# Full_lambda_init[int(len(real_cons)/2):] = -2*np.ones(int(len(real_cons)/2))
if constraint_mu_only == True: 
    nCons = int(nCons/2)
    
# else: 
#     nCons = full_con_num

Lambda_init = Full_lambda_init[:nCons]
# save that lambda init 
# ------------------------------------------


file_name_lambda =path+ 'Lambdas.csv'
with open(file_name_lambda, 'w') as new_file_lambda:
    csv_writer_lambda = csv.writer(new_file_lambda, delimiter = ',')
    csv_writer_lambda.writerow(Lambda_init)
    new_file_lambda.flush()

print(f'Saved initial lambda: of length {len(Lambda_init)}')
print(f'lambda = {Lambda_init}')
