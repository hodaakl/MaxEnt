import numpy as np 
from scipy.integrate import solve_ivp
from MomentEquations import MomentsDiff_Eq_fn

def Get_Init_fn(k): 
    """Inputs: k
    Output: z0 """
    # last two entries 
    nCom = 44
    k = 10**(k)

    k1 = k[0]*k[1]   #Synthesis of IGFR 
    k2 = k[1]   #Degredation of IGFR
    k3 = k[2]   #Binding of IGFR to IGF
    k4 = k[3]   #Unbinding IGFR to IGF 
    k5 = k[4]   #Phosphorylation of bound receptor 
    k6 = k[5]   #Dephosphorylation of bound receptor 
    k7 = k[6]   #Dephosphorylation of AKT
    k8 = k[7]   #Phosphorylation of AKT
    k9 = k[8]   #Phosphorylation of FoxO
    k10 = k[9]  #Dephosphorylation of FoxO
    k11 = k[10] #Influx of FoxO to nucleus 
    k12 = k[11] #Efflux of FoxO from nucleus
    k_tot_Akt = k[12] 
    k_tot_foxo = k[13]
    # z[0]   #R
    # z[1]   #B
    # z[2]   #P
    # z[3]   #akt
    # z[4]   #pakt
    # z[5]   #pfoxoc
    # z[6]   #foxoc
    # z[7]   #foxon

    #
    z0 = np.zeros(nCom)
    # z0_1
    z0[0]= k1/k2
    # z0_3
    z0[3] = k_tot_Akt # 34050   # total AKT which can vary 
    # z0_7 
    z0[6] = (k_tot_foxo*k12)/(k11 + k12)   #710 is total FoxO which can also vary 
    # z0_8 
    z0[7] = (k_tot_foxo*k11)/(k11 + k12)  
    # z0_1_1
    z0[8]= k1/k2
    # z0_7_7
    # -(- 710*k11^2*k12 + 710*k11*k12^2)/(k11^2*(k11 + k12))
    z0[41] = (k_tot_foxo*k11*k12)/(k11**2 + 2*k11*k12 + k12**2)
    # z0_7_8 
    z0[42] =  -(k_tot_foxo*k11*k12)/((k11 + k12)**2)
    # z0_8_8
    z0[43] = (k_tot_foxo*k11*k12)/(k11**2 + 2*k11*k12 + k12**2)
    return z0



# t_span2-tuple of floats
# Interval of integration (t0, tf). The solver starts with t=t0 and integrates until it reaches t=tf.

def solve_Moments_fn(K,IGF,t, z0):
    """Inputs: K , L = IGF , tend
    Outputs: Z solution"""
    tspan = [0, t] 
    # z0 = Get_Init_fn(K)
    sol_dyn = solve_ivp(MomentsDiff_Eq_fn, tspan, z0, method = 'BDF', args=(K,IGF))
    sol = sol_dyn.y[:,-1]
    return sol_dyn, sol
    
def FoxOn_preds_fn(k,L,t, z0):
    """"Function spits out mean and second moment of nuclear foxO at time t
    input:  L = IGF concentration in nM 
            t = end time in seconds
    output: meanfoxO, variancefoxO"""
    _, Sol_t = solve_Moments_fn(k, L, t ,z0)
    foxOn_mean, foxOn_var = Sol_t[7], Sol_t[-1]
    # get the second moment from the variance 
    # foxOn_s = foxOn_var + foxOn_mean**2
    return foxOn_mean, foxOn_var 

def cell_pred_fn(k, times_arr, L):
    """Function that gets the predictions for 6 concentrations each at 6 time points """
    nL_cons = len(L)
    nCons = int(nL_cons*len(times_arr) )
    z0 = Get_Init_fn(k)
    # get the initial conditions here
    means_pred_arr = np.zeros(nCons)
    var_pred_arr = np.zeros(nCons)
    i=0
    for igf in L: 
        # ts = 0
        for t in times_arr:

                
            
            means_pred_arr[i] , var_pred_arr[i] = FoxOn_preds_fn(k, igf, t,z0)
            i+=1
            # ts = t

    return means_pred_arr, var_pred_arr


