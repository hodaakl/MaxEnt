import csv
import numpy as np 
import random
import time
# sve_ivp
import os
import pandas as pd
from Update_LM_functions import calculate_constraints, update_lambda, openfile

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
path = spec_folder_onServer + '0227_test/'
# ------------------------------------------

file_name_lambda = path + 'Lambdas.csv'
file_name_error = path+ 'Errors.csv'
read_dictionary = np.load(spec_folder_onServer + 'Arrays_for_max_ent/cons_dict_mu_lnx_fraclr_020322.npy',allow_pickle='TRUE').item()
real_cons = read_dictionary['array']

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
Preds = calculate_constraints(data)


Error = Preds - real_cons[:len(Preds)]
Old_Lambda = Lambda_np[-1,:]
# alpha_arr = 10**(-np.log10(Preds) -1 )
ncons = len(real_cons)


#mumu_data = np.mean(real_cons[:int(ncons/4)])
#mulnx_data = np.mean(real_cons[int(ncons/4):int(ncons/2)])
#mu_per = .25
# print(mumu_data)
# print(mus_data)

#muc_len = 28 
#lnx_len = 28 
#pc_len = 28*2 
#norm_vec = np.concatenate( (mumu_data*(np.ones(muc_len)), mulnx_data*(np.ones(lnx_len))  ,  mu_per*(np.ones(pc_len)) ), axis = 0)

##
# amu = np.ones(int(ncons/4))*10**(-2) #before was 10**-1
# aln = np.ones(int(ncons/4))# before was *10
# ap = np.ones(int(ncons/2)) # *10 the lagrange multipliers are spiking 
# alpha = np.concatenate((amu, aln, ap), axis = 0)
alpha = 0.02
norm_vec = np.ones(ncons)
# alpha =np.concatenate( (np.ones(int(len(real_cons)/2))*10**(-1), np.ones(int(len(real_cons)/2))*10), axis = 0)
Lambda = update_lambda(Error = Error, old_lambda= Old_Lambda, norm_vector= norm_vec, alpha_cons = alpha)#, alpha_power = alpha_power) 

Lambda= Lambda.tolist()
Error = Error.tolist()


# ------------------------------------------
#write the result 


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

# plot the parameters 

# store the absolute relative error array 



