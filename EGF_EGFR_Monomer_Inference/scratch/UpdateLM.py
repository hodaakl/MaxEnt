# Author: Hoda Akl 
# Date : 05.03.2022 
import numpy as np 
import os
import pandas as pd
from update_lm_fns import calculate_constraints, update_lambda, openfile
from tools import write_output

## set path 
noise_factor=1
path = '/blue/pdixit/hodaakl/A1MAXENT_EGF/Percentile_NonDimerized_07152022/'
ArraysPath = '/blue/pdixit/hodaakl/A1MAXENT_EGF/Code/Percentile_Constraints/ArraysForMaxEnt/'
file_name_lambda = path + 'Lambdas.csv'
file_name_error = path+ 'Errors.csv'
if not os.path.exists(file_name_lambda): 

    raise ValueError("No saved Lagrange multipliers, nothing to update. You have to save initial Lagrange multipliers first")
# Get the Lagrange multipliers matrix 
##### 
Lambda_np = openfile(file_name_lambda)
# define the iteration
iterationp1, _ = Lambda_np.shape
iteration = iterationp1 -1
# get the predictions 
Preds_filename =path + f'cellpreds_{iteration}.csv'
df = pd.read_csv(Preds_filename, sep = ',', header = None) 
Data_np = df.to_numpy()
### take away the nan values 
idxn = np.argwhere(np.isnan(Data_np))
idx_nan_rows = idxn[:,0]
data = np.delete(Data_np,idx_nan_rows, 0)
# Get the constraints from the simulated data
# Average the simulated cells predictions
Preds = calculate_constraints(data)
ncpc = 9 # number of constraints per condition
nc = 10 # number of conditions 
nCons = int(ncpc*nc)
real_cons = .1* np.ones(nCons)
# get the errors -- Preds could be larger than constraints because other things could be saved in that file as well
Error = Preds[:nCons] - real_cons # this error is of same shape as real_cons and lambda
Old_Lambda = Lambda_np[-1,:]
ncons = len(real_cons)
# define step size 
alpha = .002
# Update the lagrange multipliers
Lambda = update_lambda(Error = Error, old_lambda= Old_Lambda, alpha_cons = alpha)#, alpha_power = alpha_power) 
Lambda= Lambda.tolist()
Error = Error.tolist()

## save the error  and the new lagrange multipliers 
write_output(file_name_error, Error )
write_output(file_name_lambda, Lambda)
### ##### SEPERATE THE PREDICTION FILE - it constaints percentiles , mu , var , parameters 
##### 
NumPars = 7 # number of parameters in this system
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
