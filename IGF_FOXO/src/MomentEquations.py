#this file defines the moment closure functions 
import numpy as np 
from scipy.integrate import solve_ivp
def MomentsDiff_Eq_fn(t, z, k ,IGF):
    """Inputs: t (in seconds), z(array of 44) , k (array of len 12), IGF (concentration in pM)
    Outputs: a list of differential equations """
    # make the exponent of the log rates 
    # define the rates 

    k = 10**(k)
    
    k1 = k[0]*k[1]   #number of receptors
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
    # define the species 
    # z_1 = z[0]   #R
    # z_2 = z[1]   #B
    # z_3 = z[2]   #P
    # z_4 = z[3]   #akt
    # z5 = z[4]   #pakt
    # z6 = z[5]   #pfoxoc
    # z7 = z[6]   #foxoc
    # z8 = z[7]   #foxon
    # zi is the mean of i
    # zii is the 2nd moment of i
    # zij is the comoment of i & j
    z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8 = z[:8]
    z_1_1, z_1_2, z_1_3, z_1_4, z_1_5, z_1_6, z_1_7, z_1_8 = z[8:16]
    z_2_2, z_2_3, z_2_4, z_2_5 , z_2_6, z_2_7, z_2_8 = z[16:23]
    z_3_3, z_3_4, z_3_5, z_3_6, z_3_7, z_3_8 = z[23:29 ]
    z_4_4, z_4_5, z_4_6, z_4_7, z_4_8 = z[29:34]
    z_5_5, z_5_6, z_5_7, z_5_8 = z[34:38]
    z_6_6, z_6_7, z_6_8 = z[38:41]
    z_7_7, z_7_8 = z[41:43]
    z_8_8 = z[43]

    # initialize the differential equation array
    ns =44
    dsyn = np.zeros(ns) 
    # write the differential equations 
    #z_1'
    dsyn[0]=k1-k2*z_1-IGF*k3*z_1+k4*z_2
    #z_2'
    dsyn[1]=IGF*k3*z_1-k2*z_2-k4*z_2-k5*z_2+k6*z_3
    #z_3'
    dsyn[2]=k5*z_2-k2*z_3-k6*z_3
    #z_4'
    dsyn[3]=-k8*z_3*z_4+k7*z_5-k8*z_3_4
    #z_5'
    dsyn[4]=k8*z_3*z_4-k7*z_5+k8*z_3_4
    #z_6'
    dsyn[5]=-k10*z_6+k9*z_5*z_7+k9*z_5_7
    #z_7'
    dsyn[6]=k10*z_6-k11*z_7-k9*z_5*z_7+k12*z_8-k9*z_5_7
    #z_8'
    dsyn[7]=k11*z_7-k12*z_8
    #z_1_1'
    dsyn[8]=k1+k2*z_1+IGF*k3*z_1+k4*z_2-2*k2*z_1_1-2*IGF*k3*z_1_1+2*k4*z_1_2
    #z_1_2'
    dsyn[9]=-IGF*k3*z_1-k4*z_2+IGF*k3*z_1_1-2*k2*z_1_2-IGF*k3*z_1_2-k4*z_1_2-k5*z_1_2+k6*z_1_3+k4*z_2_2
    #z_1_3'
    dsyn[10]=k5*z_1_2-2*k2*z_1_3-IGF*k3*z_1_3-k6*z_1_3+k4*z_2_3
    #z_1_4'
    dsyn[11]=-k8*z_4*z_1_3-k2*z_1_4-IGF*k3*z_1_4-k8*z_3*z_1_4+k7*z_1_5+k4*z_2_4
    #z_1_5'
    dsyn[12]=k8*z_4*z_1_3+k8*z_3*z_1_4-k2*z_1_5-IGF*k3*z_1_5-k7*z_1_5+k4*z_2_5
    #z_1_6'
    dsyn[13]=k9*z_7*z_1_5-k10*z_1_6-k2*z_1_6-IGF*k3*z_1_6+k9*z_5*z_1_7+k4*z_2_6
    #z_1_7'
    dsyn[14]=-k9*z_7*z_1_5+k10*z_1_6-k11*z_1_7-k2*z_1_7-IGF*k3*z_1_7-k9*z_5*z_1_7+k12*z_1_8+k4*z_2_7
    #z_1_8'
    dsyn[15]=k11*z_1_7-k12*z_1_8-k2*z_1_8-IGF*k3*z_1_8+k4*z_2_8
    #z_2_2'
    dsyn[16]=IGF*k3*z_1+k2*z_2+k4*z_2+k5*z_2+k6*z_3+2*IGF*k3*z_1_2-2*k2*z_2_2-2*k4*z_2_2-2*k5*z_2_2+2*k6*z_2_3
    #z_2_3'
    dsyn[17]=-k5*z_2-k6*z_3+IGF*k3*z_1_3+k5*z_2_2-2*k2*z_2_3-k4*z_2_3-k5*z_2_3-k6*z_2_3+k6*z_3_3
    #z_2_4'
    dsyn[18]=IGF*k3*z_1_4-k8*z_4*z_2_3-k2*z_2_4-k4*z_2_4-k5*z_2_4-k8*z_3*z_2_4+k7*z_2_5+k6*z_3_4
    #z_2_5'
    dsyn[19]=IGF*k3*z_1_5+k8*z_4*z_2_3+k8*z_3*z_2_4-k2*z_2_5-k4*z_2_5-k5*z_2_5-k7*z_2_5+k6*z_3_5
    #z_2_6'
    dsyn[20]=IGF*k3*z_1_6+k9*z_7*z_2_5-k10*z_2_6-k2*z_2_6-k4*z_2_6-k5*z_2_6+k9*z_5*z_2_7+k6*z_3_6
    #z_2_7'
    dsyn[21]=IGF*k3*z_1_7-k9*z_7*z_2_5+k10*z_2_6-k11*z_2_7-k2*z_2_7-k4*z_2_7-k5*z_2_7-k9*z_5*z_2_7+k12*z_2_8+k6*z_3_7
    #z_2_8'
    dsyn[22]=IGF*k3*z_1_8+k11*z_2_7-k12*z_2_8-k2*z_2_8-k4*z_2_8-k5*z_2_8+k6*z_3_8
    #z_3_3'
    dsyn[23]=k5*z_2+k2*z_3+k6*z_3+2*k5*z_2_3-2*k2*z_3_3-2*k6*z_3_3
    #z_3_4'
    dsyn[24]=k5*z_2_4-k8*z_4*z_3_3-k2*z_3_4-k6*z_3_4-k8*z_3*z_3_4+k7*z_3_5
    #z_3_5'
    dsyn[25]=k5*z_2_5+k8*z_4*z_3_3+k8*z_3*z_3_4-k2*z_3_5-k6*z_3_5-k7*z_3_5
    #z_3_6'
    dsyn[26]=k5*z_2_6+k9*z_7*z_3_5-k10*z_3_6-k2*z_3_6-k6*z_3_6+k9*z_5*z_3_7
    #z_3_7'
    dsyn[27]=k5*z_2_7-k9*z_7*z_3_5+k10*z_3_6-k11*z_3_7-k2*z_3_7-k6*z_3_7-k9*z_5*z_3_7+k12*z_3_8
    #z_3_8'
    dsyn[28]=k5*z_2_8+k11*z_3_7-k12*z_3_8-k2*z_3_8-k6*z_3_8
    #z_4_4'
    dsyn[29]=k8*z_3*z_4+k7*z_5+k8*z_3_4-2*k8*z_4*z_3_4-2*k8*z_3*z_4_4+2*k7*z_4_5
    #z_4_5'
    dsyn[30]=-k8*z_3*z_4-k7*z_5-k8*z_3_4+k8*z_4*z_3_4-k8*z_4*z_3_5+k8*z_3*z_4_4-k7*z_4_5-k8*z_3*z_4_5+k7*z_5_5
    #z_4_6'
    dsyn[31]=-k8*z_4*z_3_6+k9*z_7*z_4_5-k10*z_4_6-k8*z_3*z_4_6+k9*z_5*z_4_7+k7*z_5_6
    #z_4_7'
    dsyn[32]=-k8*z_4*z_3_7-k9*z_7*z_4_5+k10*z_4_6-k11*z_4_7-k8*z_3*z_4_7-k9*z_5*z_4_7+k12*z_4_8+k7*z_5_7
    #z_4_8'
    dsyn[33]=-k8*z_4*z_3_8+k11*z_4_7-k12*z_4_8-k8*z_3*z_4_8+k7*z_5_8
    #z_5_5'
    dsyn[34]=k8*z_3*z_4+k7*z_5+k8*z_3_4+2*k8*z_4*z_3_5+2*k8*z_3*z_4_5-2*k7*z_5_5
    #z_5_6'
    dsyn[35]=k8*z_4*z_3_6+k8*z_3*z_4_6+k9*z_7*z_5_5-k10*z_5_6-k7*z_5_6+k9*z_5*z_5_7
    #z_5_7'
    dsyn[36]=k8*z_4*z_3_7+k8*z_3*z_4_7-k9*z_7*z_5_5+k10*z_5_6-k11*z_5_7-k7*z_5_7-k9*z_5*z_5_7+k12*z_5_8
    #z_5_8'
    dsyn[37]=k8*z_4*z_3_8+k8*z_3*z_4_8+k11*z_5_7-k12*z_5_8-k7*z_5_8
    #z_6_6'
    dsyn[38]=k10*z_6+k9*z_5*z_7+2*k9*z_7*z_5_6+k9*z_5_7-2*k10*z_6_6+2*k9*z_5*z_6_7
    #z_6_7'
    dsyn[39]=-k10*z_6-k9*z_5*z_7-k9*z_7*z_5_6-k9*z_5_7+k9*z_7*z_5_7+k10*z_6_6-k10*z_6_7-k11*z_6_7-k9*z_5*z_6_7+k12*z_6_8+k9*z_5*z_7_7
    #z_6_8'
    dsyn[40]=k9*z_7*z_5_8+k11*z_6_7-k10*z_6_8-k12*z_6_8+k9*z_5*z_7_8
    #z_7_7'
    dsyn[41]=k10*z_6+k11*z_7+k9*z_5*z_7+k12*z_8+k9*z_5_7-2*k9*z_7*z_5_7+2*k10*z_6_7-2*k11*z_7_7-2*k9*z_5*z_7_7+2*k12*z_7_8
    #z_7_8'
    dsyn[42]=-k11*z_7-k12*z_8-k9*z_7*z_5_8+k10*z_6_8+k11*z_7_7-k11*z_7_8-k12*z_7_8-k9*z_5*z_7_8+k12*z_8_8
    #z_8_8'
    dsyn[43]=k11*z_7+k12*z_8+2*k11*z_7_8-2*k12*z_8_8
    #
    return dsyn

