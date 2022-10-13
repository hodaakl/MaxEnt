import os 
import pandas as pd 
import numpy as np 
import csv 
# ------------------------------------------
noise_factor =2
path = f'/blue/pdixit/hodaakl/A1MAXENT_EGF/Percentile_Constraints_NonDimerized/output_noise/output_ada_egf_percentile_nf_{noise_factor}_20221013/'
ArraysPath = '/blue/pdixit/hodaakl/A1MAXENT_EGF/Percentile_Constraints_NonDimerized/ArraysForMaxEnt/'
print('path is ',path)

if os.path.exists(path):
    raise ValueError("folder already exists, can't initialize lambda")
else: 
    print('creating folder')
    os.mkdir(path)# ------------------------------------------

nc = 10
ncpc = 9
nCons = int(nc*ncpc)
print('length of constraints array = ', nCons)
# real_cons = np.load(spec_folder_onServer + 'Arrays_for_max_ent/Cons_Arr_Means_SecMoment_72Scaled.npy')

lambdaZero = np.zeros(nCons)
# save that lambda init 
# ------------------------------------------
file_name_lambda =path+ 'Lambdas.csv'
with open(file_name_lambda, 'w') as new_file_lambda:
    csv_writer_lambda = csv.writer(new_file_lambda, delimiter = ',')
    csv_writer_lambda.writerow(lambdaZero)
    new_file_lambda.flush()

print(f'Saved initial lambda: of length {len(lambdaZero)}')
print(f'lambda = {lambdaZero}')
### save m and v for the adam algorithm 
m=0;v=0
np.save(path+ 'adam_m_v.npy' ,np.array([m,v]))