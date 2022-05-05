import os 
import pandas as pd 
import numpy as np 
import csv 
# ------------------------------------------
noise_factor=2
path = 'OutputFolder/'
ArraysPath = 'ArraysForMaxEnt/'
print('path is ',path)

if not os.path.exists(path): 
    os.mkdir(path)
else: 
    raise ValueError("folder already exists, can't initialize lambda")
# ------------------------------------------

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
