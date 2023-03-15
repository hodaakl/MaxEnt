from PredictionFunctions import cell_pred_fn_handover , cell_pred_fn , MomentsDiff_Eq_fn , Get_Init_fn
from PredictionFunction_v2 import cell_pred_fn_new
from scipy.integrate import solve_ivp
## 
import time 
import numpy as np
par_dict = {'par_name': ['k1', 'k2','k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k_tot_Akt', 'k_tot_foxo']
            , 'low_lim_log':np.array([2.75,-3.5,-2.25,-0.5,-0.75,-0.75,-3.5,-6.75,-6,-3.75,-2.25,-3,4.75,2])
            , 'high_lim_log': np.array([4.25,-2,-0.75,1,0.75,0.75,-2,-5.25,-4.5,-2.25,-0.75,-1.5,6.25,3.5])}
k = .5*(par_dict['low_lim_log']+par_dict['high_lim_log'])
nk = 14
# k = np.log10(np.random.rand(nk))
times_arr = np.array([ 0,  6, 12, 24, 45, 60, 90])*60 #make it in seconds 
L  = np.array([10,15,20, 25,50,250])*10**-3 #make it in nM
t0 = time.time()
xm, xv, _ = cell_pred_fn_handover(k, times_arr, L, meth = 'BDF')
print(f'hand over method took {time.time() -t0} seconds')
# print(x)
t0 = time.time()
ym, yv , _ = cell_pred_fn(k, times_arr, L, meth = 'BDF')
print(f'old method took {time.time() -t0} seconds')
# print(x)
t0 = time.time()
zm , zv , _  = cell_pred_fn_new(k, times_arr, L, meth = 'BDF')
print(f'old method took {time.time() -t0} seconds')
# print(x)
### NEW FUNCTION 
print(xv==yv)
print(xv==zv)
print(yv==zv)

print(xm==ym)
print(xm==zm)
print(ym==zm)

# print(np.where())
idx = np.where(xm!=ym)
print(xm[idx])
print(ym[idx])
# teval = times_arr
# # def solve_Moments_fn(K,IGF, t, z0, t0 = 0, meth = 'BDF'):
# #     """Inputs: K , L = IGF , tend
# #     Outputs: Z solution"""
# #     tspan = [t0, t] 
# #     # z0 = Get_Init_fn(K)
# #     sol_dyn = solve_ivp(MomentsDiff_Eq_fn, tspan, z0, method = meth, args=(K,IGF))
# #     sol = sol_dyn.y[:,-1]
# #     return sol_dyn, sol
# IGF = 25*10**-3
# z0 = Get_Init_fn(k)

# sol_dyn = solve_ivp(MomentsDiff_Eq_fn, t_span =[0, times_arr[-1]], t_eval = times_arr, y0= z0, method = 'BDF', args=(k,IGF))
# # print(sol_dyn)

# # print(times_arr)
# print(sol_dyn.y.shape)
