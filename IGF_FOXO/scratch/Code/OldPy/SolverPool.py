import numpy as np
from scipy.integrate import solve_ivp
from MomentEquations import MomentsDiff_Eq_fn
from PredictionFunctions import FoxOn_preds_fn

L  = np.array([10,15,20,50])*10**-3 #make it in nM
times_arr = np.array([ 0,  6, 12, 24, 45, 60, 90])*60 #make it in seconds 
