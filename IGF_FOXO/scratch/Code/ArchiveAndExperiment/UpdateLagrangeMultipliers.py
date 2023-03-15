import csv
import numpy as np 
import random
import time
# sve_ivp
import os
import pandas as pd
from Update_LM_functions import calculate_constraints, update_lambda, openfile
from Write_Output_fn import write_output

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
foldername = '0413_Constrained/'
path = spec_folder_onServer + foldername 

# where are you saving 
# Arrays path 
arrays_path = spec_folder_onServer + 'Arrays_for_max_ent/'
# where are you saving 
save_locally=False
if save_locally == True: 
    path = foldername
    arrays_path = '/Users/hodaakl/Documents/github/MaxEnt_FoxO/Arrays_for_max_ent/'
# ------------------------------------------
# ------------------------------------------

file_name_lambda = path + 'Lambdas.csv'
file_name_error = path+ 'Errors.csv'
# read_dictionary = np.load(spec_folder_onServer + 'Arrays_for_max_ent/cons_dict_mu_lnx_fraclr_020322.npy',allow_pickle='TRUE').item()

if not os.path.exists(file_name_lambda): 

    raise ValueError("No saved Lagrange multipliers, nothing to update. You have to save initial Lagrange multipliers first")

Lambda_np = openfile(file_name_lambda)

iterationp1, _ = Lambda_np.shape
iteration = iterationp1 -1

#
moments_filename =path + f'cellpreds_{iteration}.csv'


df = pd.read_csv(moments_filename, sep = ',', header = None) 

Data_np = df.to_numpy()
### take away the nan values 
idxn = np.argwhere(np.isnan(Data_np))
idx_nan_rows = idxn[:,0]

data = np.delete(Data_np,idx_nan_rows, 0)
Preds = calculate_constraints(data) ## this could be longer the number of constraints 
ncpc = 9 # 
nc = 28 
nCons = int(ncpc*nc)
# read_dictionary = np.load(spec_folder_onServer + 'Arrays_for_max_ent/cons_dict_mu_lnx_fraclr_020322.npy',allow_pickle='TRUE').item()
pf=1
real_cons = .1* np.ones(nCons)*pf

Error = Preds[:nCons] - real_cons
Old_Lambda = Lambda_np[-1,:]
# alpha_arr = 10**(-np.log10(Preds) -1 )
ncons = len(real_cons)

alpha = .5
norm_vec = np.ones(ncons)
# alpha =np.concatenate( (np.ones(int(len(real_cons)/2))*10**(-1), np.ones(int(len(real_cons)/2))*10), axis = 0)
Lambda = update_lambda(Error = Error, old_lambda= Old_Lambda, alpha_cons = alpha)#, alpha_power = alpha_power) 

Lambda= Lambda.tolist()
Error = Error.tolist()


write_output(file_name_error, Error )

#################################################### Storing the Lambda  

if os.path.exists(file_name_lambda): 
    with open(file_name_lambda, 'a') as add_file_lambda:
        csv_adder_lambda = csv.writer(add_file_lambda, delimiter = ',')
        csv_adder_lambda.writerow(Lambda)
        add_file_lambda.flush()
else:
    print('trouble loading file')
### ##### SEPERATE THE PREDICTION FILE - it constaints percentiles , mu , var , parameters 
##### 
NumPars = 14 # number of parameters in this system
par_data = data[:,-NumPars:]
par_filename = path + f'/params_{iteration}.csv' 
# dump the numpy matrix into a csv file 
np.savetxt(par_filename, par_data, delimiter=",")
## 

############################################# Plotting
import matplotlib.pyplot as plt
if os.path.isdir(path+'figs')==False:
    os.mkdir(path+'figs') 
# plot the absolute relative error
df_err = pd.read_csv(file_name_error, sep = ',', header = None) 
err_np = df_err.to_numpy()
rc_m= np.tile(real_cons[:len(err_np[0,:])] , [err_np.shape[0],1])
#print(rc_m.shape)
mean_err = np.mean(abs(err_np), axis = 1)
# mean_err.shape
real_abs = abs(err_np/rc_m)
mean_rel_abs = np.mean(real_abs, axis = 1)
# mean_err = np.mean(abs(err_np), axis = 1)
plt.plot(range(len(mean_rel_abs) ), mean_rel_abs)
plt.ylabel('Mean Abs relative Error ')
plt.xlabel('Iteration')
# plt.title('Iteration[1:]')
plt.savefig(path+'figs/error.png')
print('saved figure')
# save error array 
np.save(path+'mean_abs_error.npy', mean_rel_abs)
# plot the lagrange multipliers 
