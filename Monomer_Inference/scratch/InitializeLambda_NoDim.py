#!/usr/bin/env python
# coding: utf-8

# In[3]:


import csv
import numpy as np 
import random
import time
from scipy.integrate import solve_ivp
from collections import defaultdict
import os
outputfolder = '/blue/pdixit/hodaakl/output/NoDim_Run1_0308/'
Constraint_mu_only = False 


if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)


    

file_name_lambda =outputfolder+ 'Lambdas.csv'

if Constraint_mu_only: 
    dim = 10
    Lambda_init = 0.001*np.random.rand(dim) 

else: 
    dim = 20
    Lambda_init = np.random.rand(dim)
    
    Lambda_init[:10] =Lambda_init[:10]*10**(-5)  ## This then is of order 10^-6
    Lambda_init[10:] =Lambda_init[10:]*10**(-10)  # I want this to be of order 10^-11



if os.path.exists(file_name_lambda): 
    print('file already exists')
    
else:
    
    with open(file_name_lambda, 'w') as new_file_lambda:
        csv_writer_lambda = csv.writer(new_file_lambda, delimiter = ',')
        csv_writer_lambda.writerow(Lambda_init)
        new_file_lambda.flush()


# In[4]:





# In[ ]:




