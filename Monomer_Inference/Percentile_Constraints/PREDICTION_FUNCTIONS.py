## Author: Hoda Akl 
## Date Created: 05/04/2022
#
# File contains functions that obtains the means and the second moments of the EGF-EGFR momoner model. 
# Equations are solved analytically using Moment Closure Method 
# 
import numpy as np
from scipy.stats import gamma


def segfr_preds(L,  pars_arr,n=1, SF = 0.00122 ):
    """ Obtains steady state predictions: the EGFR receptor numbers for a given Ligand concentration and Parameters vector 
    Inputs : L -> Ligand Concentration in nM 
            pars_arr -> Parameter vector : [ksyn, k1, kn1, kap, kdp, ki, kis]
                Pars array is of length 7 
            n -> noise factor 
            SF -> Conversion factor to change a.u. to molecules 
                The SF is to change from a.u. to molecules based on 305 a.u. = 250*10^3 molecules  )
    Outputs: mean and second moment after noise adjustmenet of the distribution of EGFR for that ligand concentration."""
    # idx = [16, 0, 1, 4, 5, 8, 9]
    [ksyn, k1, kn1, kap, kdp, ki, kis] = 10**pars_arr # parameters are in log scale!  
    ksyn = ksyn/SF
#     kap = kap/SF
    R0  = ksyn/ki
    mean= (ki* (kap* (kis+k1* L)+(kis+kdp) *(ki+kn1+k1* L))* R0)/(kap *kis* (ki+k1 *L)+ki *(kis+kdp) *(ki+kn1+k1*L))
    sec_mom = (ki*(kap *(kis+k1*L)+(kis+kdp)* (ki+kn1+k1*L)) *R0* (kap* kis* (ki+k1* L)+kap* ki* (kis+k1* L) *R0+ki* (kis+kdp)* (ki+kn1+k1* L)* (1+R0)))/(kap* kis* (ki+k1*L)+ki* (kis+kdp) *(ki+kn1+k1 *L))**2
    sec_mom_new = (sec_mom - mean**2)*(n**2) + mean**2 
    return mean, sec_mom_new

def cell_pred_fn(pars, Ligand_conc, n=1 ):
    """solves for all the ligand concentrations and returns the means and the second moments
    Inputs: pars: Parameter vector : [ksyn, k1, kn1, kap, kdp, ki, kis]
                Pars array is of length 7 
            Ligand_conc : array of ligand concentrations in nM  """
    conc_len = len(Ligand_conc)
    means_arr = np.zeros(conc_len)
    secmom_arr = np.zeros(conc_len)
    for i in range(conc_len): 
        L = Ligand_conc[i]
        mu, s = segfr_preds(L = L , pars_arr = pars,SF = 0.00122 , n=n)
        means_arr[i] = mu
        secmom_arr[i] = s
    return means_arr, secmom_arr


def get_prec_preds(Bounds_Perc_Dict, k, L, n=1): 
    """" Gets the percentile prediction for one cell 
    Inputs: Bounds_Perc_Dict: Dictionary of the bounds that produce specific percentiles
            k: parameter vector 
            L: Ligand concentration for solving the system 
             
    Output: PredArr : probability that the cell lies in a specific bin --> [bin1_L1, bin1_L2, bin1_L3..... ......BinN_L9, BinN_L10] # length 90
            means: means of the distributions length 10 
            secmoms: second moments length 10
            vars : variance length 10 
    """

    means, secmoms = cell_pred_fn(pars = k, Ligand_conc = L,n=n)
    vars = secmoms - means**2
    alpha_arr = (means**2)/(vars)
    scale_arr = vars/means
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
        #the output means [bin1_L1, bin1_L2, bin1_L3..... ......BinN_L9, BinN_L10]
        PredArr = np.concatenate((PredArr, spp), axis = 0)
    return PredArr , means , secmoms, vars 