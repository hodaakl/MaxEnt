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
outputfolder = '/blue/pdixit/hodaakl/output/MaxEnt_0210/Run1/'
file_name_lambda =outputfolder+ 'Lambdas.csv'
if os.path.exists(file_name_lambda): 
    print('file already exists')
    
else:
    Lambda = np.zeros(24)
    with open(file_name_lambda, 'w') as new_file_lambda:
        csv_writer_lambda = csv.writer(new_file_lambda, delimiter = ',')
        csv_writer_lambda.writerow(Lambda)
        new_file_lambda.flush()