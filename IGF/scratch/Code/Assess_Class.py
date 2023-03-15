import numpy as np
import os 
import pandas as pd


class SimLoader:
    def __init__(self, output_dir):
        """ Loads the output directory, lambdas and determines the latest index """

        if os.path.exists(output_dir)==False: 
            raise ValueError('Path does not exist')
        if output_dir[-1] != '/':
            output_dir = f'{output_dir}/'
        self.directory = output_dir
        df= pd.read_csv(output_dir + 'Lambdas.csv', sep = ',', header = None)
        self.lambda_mat =  df.to_numpy()
        df= pd.read_csv(output_dir + 'Errors.csv', sep = ',', header = None)
        self.error_mat =  df.to_numpy()
        self.latest_iter_idx =  self.lambda_mat.shape[0] -2

    def get_mean_rel_err(self):
        ErrorMat = self.error_mat
        
    
    def get_best_iter(self):

