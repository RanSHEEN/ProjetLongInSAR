import numpy as np
import scipy.linalg as La
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0,
    os.path.join(os.path.dirname(__file__), '../'))

from src.data_generation import simulate_Covariance, phasegeneration, \
    simulate_gaussiandata, simulate_scaledgaussiandata
from src.covariance_estimators import SCM, regul_linear, bandw, SCM_LR, \
    tyler_estimator_covariance, tyler_estimator_covariance_LR, corr_phase, corr_phase_LR
from src.optimization import MM_KL_IPL, MM_LS_IPL, RG_comet_IPL, RG_LS_IPL

def data_gen(size=200, p=30, n=64, cost='LS', estimator='PO', regul='SK'):
    rho_list = np.random.uniform(low=0.96, high=0.99, size=200)
    b = 3 # bandwidth parameter
    alpha = 0.5 # coefficient regularization
    rank = 1 # rank of the covariance matrix (p if full-rank)
    phasechoice = 'random'
    nu = 0

    delta_thetasim_list = []
    SigmaTrue_list = []
    trueCov_list = []
    X_list = []
    Sigma_tilde_list = [] 
    w_theta_list = []

    for rho in rho_list:
        #Génération des matrices de covariance estimées sigma_tilde
        delta_thetasim = phasegeneration(phasechoice,p) #generate phase with either random or linear. for linear, define last phase is needed
        SigmaTrue = La.toeplitz(rho**np.arange(p))
        trueCov = simulate_Covariance(SigmaTrue, delta_thetasim)

        delta_thetasim_list.append(delta_thetasim)
        SigmaTrue_list.append(SigmaTrue)
        trueCov_list.append(trueCov)

        if nu == 0: 
            X = simulate_gaussiandata(trueCov, p,n)
        else:
            X = simulate_scaledgaussiandata(trueCov, nu, p,n)

        X_list.append(X)

        if estimator == 'SCM':
            Sigma = SCM(X)
            if regul == 'LR':
                Sigma_tilde = SCM_LR(X,rank)
            if regul == 'SK': 
                Sigma_tilde = regul_linear(Sigma,alpha)
            if regul == 'BW':
                Sigma_tilde = bandw(Sigma,b)
            if regul == False:
                Sigma_tilde = Sigma

        if estimator == 'PO':
            Sigma = corr_phase(X)
            if regul == 'LR':
                Sigma_tilde = corr_phase_LR(X,rank)
            if regul == 'SK': 
                Sigma_tilde = regul_linear(Sigma,alpha)
            if regul == 'BW':
                Sigma_tilde = bandw(Sigma,b)
            if regul == False:
                Sigma_tilde = Sigma
        Sigma_tilde_list.append(Sigma_tilde)
        w_theta = RG_LS_IPL(Sigma_tilde, 100, True, False, False)
        w_theta_list.append(w_theta)
    
    return X_list, w_theta_list