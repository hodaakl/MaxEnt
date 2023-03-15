import numpy as np 
from scipy.integrate import solve_ivp
from MomentEquations import MomentsDiff_Eq_fn
from scipy.stats import gamma 

def Get_Init_fn(k): 
    """Inputs: k
    Output: z0 """
    # last two entries 
    nCom = 44
    k = 10**(k)

    k1 = k[0]*k[1]   #Synthesis of IGFR 
    k2 = k[1]   #Degredation of IGFR
    # k3 = k[2]   #Binding of IGFR to IGF
    # k4 = k[3]   #Unbinding IGFR to IGF 
    # k5 = k[4]   #Phosphorylation of bound receptor 
    # k6 = k[5]   #Dephosphorylation of bound receptor 
    # k7 = k[6]   #Dephosphorylation of AKT
    # k8 = k[7]   #Phosphorylation of AKT
    # k9 = k[8]   #Phosphorylation of FoxO
    # k10 = k[9]  #Dephosphorylation of FoxO
    k11 = k[10] #Influx of FoxO to nucleus 
    k12 = k[11] #Efflux of FoxO from nucleus
    k_tot_Akt = k[12] 
    k_tot_foxo = k[13]

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



def solve_Moments_fn(K,IGF, t, z0, t0 = 0, meth = 'BDF'):
    """Inputs: K , L = IGF , tend
    Outputs: Z solution"""
    tspan = [t0, t] 
    # z0 = Get_Init_fn(K)
    sol_dyn = solve_ivp(MomentsDiff_Eq_fn, tspan, z0, method = meth, args=(K,IGF))
    sol = sol_dyn.y[:,-1]
    return sol_dyn, sol
    
def FoxOn_preds_fn(k,L,t,  z0, t0 = 0, meth = 'BDF'):
    """"Function spits out mean and second moment of nuclear foxO at time t
    input:  L = IGF concentration in nM 
            t = end time in seconds
    output: meanfoxO, variancefoxO"""
    _, Sol_t = solve_Moments_fn(k, L, t , z0, t0 = t0, meth=meth)
    foxOn_mean, foxOn_var = Sol_t[7], Sol_t[-1]
    # get the second moment from the variance 
    # foxOn_s = foxOn_var + foxOn_mean**2
    return foxOn_mean, foxOn_var , Sol_t

def cell_pred_fn_handover(k, times_arr, L, meth = 'BDF'):
    """Function that gets the predictions for 6 concentrations each at 6 time points """
    nL_cons = len(L)
    nCons = int(nL_cons*len(times_arr) )
    z0 = Get_Init_fn(k)
    # get the initial conditions here
    means_pred_arr = np.zeros(nCons)
    var_pred_arr = np.zeros(nCons)
    i=0
    for igf in L: 
        init_cond = z0.copy()
        # ts = 0
        for ti in range(len(times_arr)):
            t = times_arr[ti]
            if t==0:
                t0 = 0
            else :
                t0 = times_arr[ti-1]
            means_pred_arr[i] , var_pred_arr[i] , init_cond = FoxOn_preds_fn(k, igf, t,init_cond, t0 = t0, meth=meth)
            i+=1
            # ts = t

    return means_pred_arr, var_pred_arr , z0[7]


def cell_pred_fn(k, times_arr, L, meth = 'BDF'):
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

                
            
            means_pred_arr[i] , var_pred_arr[i] , _= FoxOn_preds_fn(k, igf, t,z0, meth=meth)
            i+=1
            # ts = t

    return means_pred_arr, var_pred_arr , z0[7]

# def get_percentiles 
def get_prec_preds(Bounds_Perc_Dict, k, times_arr, L, meth  = 'BDF' ): 
    """" Gets the percentile prediction for one cell 
    Inputs: Bounds_Perc_Dict: Dictionary of the bounds that produce specific percentiles
            k: parameter vector 
            times_arr: times for solving the differential equation 
            L: Ligand concentration for solving the differential equations 
             
    """

    means, var , init_point= cell_pred_fn_handover(k, times_arr, L, meth = meth)
    alpha_arr = (means**2)/(var)
    scale_arr = var/means
    sumspp = np.zeros(len(alpha_arr))
    PredArr = np.array([])
    firstkey = list(Bounds_Perc_Dict.keys())[0]

    for key in Bounds_Perc_Dict: 
        # print(key)
        bound = Bounds_Perc_Dict[key]
        if key==firstkey:
        
            spp = gamma.cdf(bound, a =alpha_arr, scale = scale_arr)#specific percentile array 
            sumspp+=spp
        else:
            spp= gamma.cdf(bound, a =alpha_arr, scale = scale_arr) -sumspp
            sumspp+=spp#specific percentile array 
        # print(spp)  #     
        #the output means [percentile_L1_T1_bin1, percentile_L1_T2_bin1, percentile_L1_T3_bin1..... ...
        # .... percentile_L4_T6_bin9... percentile_L4_T7_bin9]
        # 
        PredArr = np.concatenate((PredArr, spp), axis = 0)
    return PredArr , means , var , init_point

